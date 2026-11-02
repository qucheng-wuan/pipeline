"""CPU trace bookkeeping and strict comparison; no model/GPU execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from transparent_depth.stage1 import PROJECT


def module(name):
    spec=importlib.util.spec_from_file_location(name,PROJECT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class TraceTests(unittest.TestCase):
    def test_dtype_normalization_sha_and_failed_numeric_status(self):
        trace=module('trace_stage1_live');comparison=module('compare_stage1_traces')
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b'
            for directory,tensor in [(a,torch.ones(2,3)),(b,np.ones((2,3),dtype=np.float32))]:
                recorder=trace.Recorder(directory);recorder.put('final_depth',tensor)
                trace.write_json(directory/'complete.json',{'status':'trace_complete','rgb_sha256':'a'*64,
                    'calls':{'moge':2},'fields':recorder.fields})
            self.assertEqual(comparison.compare(a,b)['status'],'passed')
            record=trace.Recorder(b);record.put('final_depth',np.full((2,3),1.001,dtype=np.float32))
            trace.write_json(b/'complete.json',{'status':'trace_complete','rgb_sha256':'a'*64,'calls':{'moge':2},'fields':record.fields})
            failed=comparison.compare(a,b);self.assertEqual(failed['status'],'failed_numeric_comparison')
            self.assertEqual(failed['fields']['final_depth']['outside_tolerance'],6)
            self.assertIs(failed['tolerance_automatically_relaxed'],False)
            path=b/record.fields['final_depth']['path'];path.write_bytes(b'tampered')
            with self.assertRaises(ValueError):comparison.compare(a,b)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bfloat16_is_losslessly_recorded_as_float32(self):
        trace=module('trace_stage1_live')
        with tempfile.TemporaryDirectory() as folder:
            record=trace.Recorder(folder);value=torch.tensor([1.,1.25],dtype=torch.bfloat16)
            record.put('bf16',value);field=record.fields['bf16']
            self.assertEqual(field['source_dtype'],'bfloat16');self.assertEqual(field['stored_dtype'],'float32')
            self.assertTrue(np.array_equal(np.load(Path(folder)/field['path']),value.float().numpy()))


if __name__=='__main__':unittest.main()
