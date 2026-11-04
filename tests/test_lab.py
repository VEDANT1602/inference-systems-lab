import copy
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from inference_lab.benchmark import (load_workload, quantile, token_cost,
                                     economic_break_even, run_batch, compare_release)
from inference_lab.fixture import FixtureServer, original_cases
from inference_lab.protocol import Endpoint, ProtocolError, sse_events, stream_chat, quality
from inference_lab.report import link, render, write_json
from inference_lab.kv import TinyAttention, kv_benchmark
from inference_lab.contract_gate import grounded_json_gate
import torch


class StreamParserTests(unittest.TestCase):
    def test_every_byte_boundary_and_utf8(self):
        wire = ': caf\u00e9\r\nid: 1\r\ndata: {"x":1}\r\n\r\ndata: [DONE]\r\n\r\n'.encode()
        self.assertEqual(list(sse_events(bytes([x]) for x in wire)), ['{"x":1}', '[DONE]'])

    def test_multiline_data_and_comments(self):
        self.assertEqual(list(sse_events([b': ping\ndata: one\ndata: two\n\n'])), ['one\ntwo'])

    def test_cr_only_and_mixed_line_endings(self):
        self.assertEqual(list(sse_events([b'data: a\r', b'\rdata: b\r\n\n'])), ['a', 'b'])

    def test_partial_event_rejected(self):
        with self.assertRaises(ProtocolError):
            list(sse_events([b'data: incomplete\n']))

    def test_response_size_and_bad_utf8(self):
        with self.assertRaises(ProtocolError):
            list(sse_events([b'data: hello\n\n'], max_bytes=2))
        with self.assertRaises(UnicodeError):
            list(sse_events([b'data: \xff\n\n']))

    def test_exact_schema_and_finish_required(self):
        self.assertFalse(quality('{"label":"billing","extra":1}', 'billing', 'stop')[0])
        self.assertFalse(quality('{"label":[]}', 'billing', 'stop')[0])
        self.assertFalse(quality('{"label":"billing"}', 'billing', 'length')[0])
        self.assertTrue(quality('{"label":"billing"}', 'billing', 'stop')[0])

    def test_deep_answer_json_is_rejected_without_escaping(self):
        result = quality('['*1100+']'*1100, 'billing', 'stop')
        self.assertEqual(result, (False, 'invalid_json', None))

    def test_endpoint_and_remote_execution_guards(self):
        for value in ('http://example.com/v1', 'https://user:key@example.com/v1',
                      'https://example.com/v1?token=secret', 'file:///tmp/model'):
            with self.assertRaises(ValueError):
                Endpoint('bad', value, 'model')
        remote = Endpoint('remote', 'https://example.com/v1', 'model', 'sensitive')
        self.assertNotIn('sensitive', repr(remote))
        with self.assertRaises(ValueError):
            stream_chat(remote, original_cases()[0])


class HTTPIntegrationTests(unittest.TestCase):
    cases = original_cases()

    def request(self, mode='correct', timeout=2):
        with FixtureServer(self.cases, mode=mode) as fixture:
            return stream_chat(fixture.endpoint, self.cases[0], timeout_s=timeout)

    def test_normal_real_socket_stream_and_secret_redaction(self):
        with FixtureServer(self.cases) as fixture:
            secured = Endpoint('test', fixture.endpoint.base_url, 'known-answer-fixture', 'secret-key-123')
            result = stream_chat(secured, self.cases[0])
        self.assertTrue(result['transport_success'])
        self.assertTrue(result['quality_pass'])
        self.assertGreater(result['content_chunks'], 1)
        self.assertEqual(len(result['content_chunk_gap_ms']), result['content_chunks'] - 1)
        self.assertIsNone(result['inter_token_latency_ms'])
        self.assertEqual(result['usage'], {'prompt_tokens':64, 'completion_tokens':6})
        self.assertNotIn('secret-key-123', json.dumps(result))

    def test_wrong_label_is_completed_but_rejected(self):
        result = self.request('wrong_label')
        self.assertTrue(result['transport_success'])
        self.assertFalse(result['quality_pass'])
        self.assertEqual(result['quality_reason'], 'wrong_label')

    def test_invalid_json_quality_failure(self):
        result = self.request('invalid_json')
        self.assertTrue(result['transport_success'])
        self.assertFalse(result['quality_pass'])
        self.assertEqual(result['quality_reason'], 'invalid_json')

    def test_http_error_retains_denominator(self):
        result = self.request('http_error')
        self.assertEqual(result['error'], 'http_429')
        self.assertFalse(result['transport_success'])
        self.assertIsNone(result['usage'])

    def test_timeout_is_bounded(self):
        result = self.request('timeout', .04)
        self.assertEqual(result['error'], 'timeout')
        self.assertLess(result['e2e_ms'], 200)

    def test_trickled_headers_cannot_extend_deadline(self):
        result = self.request('trickle_headers', .05)
        self.assertEqual(result['error'], 'timeout')
        self.assertLess(result['e2e_ms'], 250)

    def test_trickled_body_cannot_extend_deadline(self):
        result = self.request('trickle_body', .05)
        self.assertEqual(result['error'], 'timeout')
        self.assertLess(result['e2e_ms'], 250)

    def test_deep_sse_json_preserves_batch_failures(self):
        with FixtureServer(self.cases, mode='deep_event') as fixture:
            result = run_batch(self.cases[:2], fixture.endpoint)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['quality_successes'], 0)
        self.assertEqual(result['transport_successes'], 0)
        self.assertEqual(result['errors_by_type'], {'protocol_error':2})

    def test_deep_answer_json_preserves_transport_and_quality_counts(self):
        with FixtureServer(self.cases, mode='deep_answer') as fixture:
            result = run_batch(self.cases[:2], fixture.endpoint)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['transport_successes'], 2)
        self.assertEqual(result['quality_successes'], 0)
        self.assertTrue(all(t['attempts'][0]['quality_reason'] == 'invalid_json' for t in result['traces']))

    def test_missing_usage_does_not_mean_free(self):
        result = self.request('no_usage')
        self.assertTrue(result['quality_pass'])
        self.assertIsNone(result['usage'])

    def test_bool_usage_rejected(self):
        result = self.request('bad_usage')
        self.assertEqual(result['error'], 'protocol_error')
        self.assertFalse(result['transport_success'])

    def test_interrupted_stream_rejected(self):
        self.assertEqual(self.request('interrupted')['error'], 'protocol_error')

    def test_fallback_retains_cost_and_primary_time(self):
        with FixtureServer(self.cases, mode='wrong_label') as primary, FixtureServer(self.cases) as fallback:
            result = run_batch(self.cases[:2], primary.endpoint, fallback=fallback.endpoint,
                               input_per_million=1, output_per_million=2,
                               fallback_input_per_million=3, fallback_output_per_million=4)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['attempts'], 4)
        self.assertEqual(result['quality_successes'], 2)
        self.assertAlmostEqual(result['cost']['total_cost'], 2 * ((64+12)+(192+24))/1e6)
        trace = result['traces'][0]
        self.assertGreaterEqual(trace['accepted_ttft_proxy_ms'], trace['attempts'][0]['e2e_ms'])

    def test_all_failures_have_null_success_latency_and_cost(self):
        with FixtureServer(self.cases, mode='http_error') as fixture:
            result = run_batch(self.cases[:2], fixture.endpoint, input_per_million=1, output_per_million=2)
        self.assertEqual(result['quality_success_rate'], 0)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['errors_by_type'], {'http_429':2})
        self.assertIsNone(result['ttft_proxy_successes_ms']['p95'])
        self.assertIsNone(result['cost']['total_cost'])
        self.assertIsNone(result['cost']['cost_per_quality_success'])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_json(directory/'metrics.json', {'endpoint_run':result, 'attention_run':None,
                       'execution_kind':'loopback_protocol_fixture'})
            render(directory)
            report = (directory/'case-study.md').read_text(encoding='utf-8')
            self.assertIn('N/A', report)
            self.assertIn('0/2', report)


class AccountingTests(unittest.TestCase):
    def test_quantile_uses_observations_and_empty_is_null(self):
        self.assertEqual(quantile([0, 10], .95), 9.5)
        self.assertIsNone(quantile([], .5))
        with self.assertRaises(ValueError):
            quantile([float('nan')], .5)

    def test_token_cost_and_crossing_units(self):
        self.assertEqual(token_cost({'prompt_tokens':1_000_000,'completion_tokens':2_000_000}, 2, 3), 8)
        self.assertIsNone(token_cost(None, 2, 3))
        self.assertAlmostEqual(economic_break_even(3.6, .001), 1)
        self.assertIsNone(economic_break_even(1, 0))
        for value in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                token_cost(None, value, 1)
            with self.assertRaises(ValueError):
                economic_break_even(value, None)

    def test_missing_fallback_prices_rejected_before_network(self):
        local = Endpoint('local', 'http://127.0.0.1:1/v1', 'model')
        with self.assertRaises(ValueError):
            run_batch(original_cases()[:1], local, fallback=local,
                      input_per_million=1, output_per_million=1)

    def test_release_comparison_rejects_unmatched_contract(self):
        a = {'workload_sha256':'a', 'requests':12, 'concurrency':2, 'execution_kind':'fixture',
             'quality_success_rate':1, 'e2e_all_requests_ms':{'p95':10}}
        b = copy.deepcopy(a)
        self.assertEqual(compare_release(a,b)['decision'], 'candidate_passes_declared_gates')
        b['quality_success_rate'] = .9
        self.assertEqual(compare_release(a,b)['decision'], 'hold_candidate')
        for field, value in [('workload_sha256','b'), ('concurrency',3), ('execution_kind','endpoint')]:
            b = copy.deepcopy(a)
            b[field] = value
            with self.assertRaises(ValueError):
                compare_release(a,b)

    def test_workload_exact_schema_unique_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'workload.jsonl'
            cases = original_cases()
            path.write_text('\n'.join(json.dumps(x) for x in cases), encoding='utf-8')
            self.assertEqual(len(load_workload(path)), 12)
            path.write_text(json.dumps(cases[0])+'\n'+json.dumps(cases[0]), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_workload(path)

    def test_arbitrary_report_directory_links_resolve(self):
        source = Path(__file__).resolve().parents[1]/'data'/'DATA_CARD.md'
        with tempfile.TemporaryDirectory() as temp:
            relative = link(source, Path(temp))
            self.assertEqual((Path(temp)/relative).resolve(), source.resolve())

    def test_notebook_has_compilable_output_free_cells(self):
        path = Path(__file__).resolve().parents[1]/'notebooks'/'free_gpu_lab.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                ast.parse(''.join(cell['source']))


class AttentionTests(unittest.TestCase):
    def test_cached_matches_full_for_multiple_shapes(self):
        for prefix, decode in ((1,1), (8,5), (64,24)):
            result = kv_benchmark(prefix=prefix, decode=decode, repeats=2, warmup=1)
            self.assertTrue(result['correctness']['allclose'])
            self.assertGreater(result['median_full_recompute_ms'], 0)
            self.assertEqual(result['device'], 'cpu')

    def test_future_tokens_do_not_change_past_outputs(self):
        torch.manual_seed(2)
        model = TinyAttention().eval()
        x = torch.randn(1,5,64)
        changed = x.clone()
        changed[:,3:] *= 100
        with torch.inference_mode():
            self.assertTrue(torch.allclose(model.full(x)[:,:3], model.full(changed)[:,:3]))

    def test_decode_requires_exactly_one_position(self):
        model = TinyAttention()
        x = torch.randn(1,4,64)
        with self.assertRaises(ValueError):
            model.decode(x[:,:2], model.prefill(x[:,:1]))

    def test_device_failure_is_explicit(self):
        with self.assertRaises(ValueError):
            kv_benchmark(device='cpu', dtype_name='float16')
        if not torch.cuda.is_available():
            self.assertEqual(kv_benchmark(device='cuda')['status'], 'not_run_cuda_unavailable')


class GroundedExtractionTests(unittest.TestCase):
    def test_deep_candidate_json_is_a_failure(self):
        result = grounded_json_gate('['*1100+']'*1100, '30 days', {'notice_period':'30 days'})
        self.assertEqual(result, {'pass':False, 'reason':'invalid_json'})
    def test_original_five_case_workload_can_pass_grounded_gold(self):
        path = Path(__file__).resolve().parents[1]/'data'/'contract_extraction.jsonl'
        cases = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(cases), 5)
        for case in cases:
            output = {name:{'value':value, 'quote':value} for name,value in case['expected'].items()}
            self.assertTrue(grounded_json_gate(json.dumps(output), case['source'], case['expected'])['pass'])
    def test_exact_quote_with_correct_value(self):
        source = 'The agreement provides 30 days notice.'
        output = json.dumps({'notice_period':{'value':'30 days','quote':'30 days notice'}})
        self.assertTrue(grounded_json_gate(output, source, {'notice_period':'30 days'})['pass'])

    def test_invented_quote_and_grounded_but_wrong_field_fail(self):
        source = 'Renewal occurs every 30 days. Termination requires 90 days notice.'
        for field, quote in [('30 days','every 30 days'), ('90 days','90 days prior written notice')]:
            output = json.dumps({'notice_period':{'value':field,'quote':quote}})
            self.assertFalse(grounded_json_gate(output, source, {'notice_period':'90 days'})['pass'])

    def test_missing_field_and_invalid_json_fail(self):
        for output in ('bad', '{}', '{"notice_period": "30 days"}'):
            self.assertFalse(grounded_json_gate(output, '30 days', {'notice_period':'30 days'})['pass'])

    def test_explicit_absence_is_valid_only_for_missing_gold(self):
        output = '{"notice_period":{"value":null,"quote":null}}'
        self.assertTrue(grounded_json_gate(output, 'No notice period provided.', {'notice_period':None})['pass'])
        self.assertFalse(grounded_json_gate(output, '30 days notice', {'notice_period':'30 days'})['pass'])


if __name__ == '__main__':
    unittest.main()
