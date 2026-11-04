"""Stage verified model/statistics copies; original training files stay frozen."""

import argparse
import json
import shutil
from pathlib import Path

from data_pipeline.normalization import DEFAULT_STATS,file_sha256,metadata_path
from data_pipeline.preprocessing import FEATURE_NAMES,RECIPE
from inference.storage import canonical
from optimisation.benchmark import read_registered_models
from training.dataset_io import load_dataset


def prepare(args):
    arrays,meta=load_dataset(args.dataset_dir,args.stats)
    specs,protected,validation_hash=read_registered_models(arrays,meta,args.models_dir)
    protected.extend([args.stats,metadata_path(args.stats),args.dataset_dir/'logibridge_dataset.npz'])
    before={str(p):file_sha256(p) for p in protected}
    candidates={spec['variant']:spec for spec in specs}
    if args.default_variant not in candidates:raise ValueError("Unknown default variant")
    folder=args.output_dir;models=folder/'models';models.mkdir(parents=True,exist_ok=True)
    models.chmod(0o755)
    registry={"schema_version":1,"feature_names":list(FEATURE_NAMES),"recipe":RECIPE,
              "normalization_stats_sha256":meta['normalization_stats_sha256'],
              "normalization_metadata_sha256":file_sha256(metadata_path(args.stats)),
              "dataset_sha256":meta['dataset_sha256'],"validation_indices_sha256":validation_hash,
              "models":[{"variant":s['variant'],"sha256":file_sha256(s['path']),
                         "bytes":s['path'].stat().st_size} for s in specs]}
    for spec in specs:
        target=models/(spec['variant']+'.tflite')
        shutil.copyfile(spec['path'],target);target.chmod(0o644)
    shutil.copyfile(args.stats,folder/'training_stats.npy')
    shutil.copyfile(metadata_path(args.stats),folder/'training_stats.meta.json')
    (folder/'runtime_registry.json').write_text(canonical(registry)+'\n',encoding='utf-8')
    shutil.copyfile(candidates[args.default_variant]['path'],folder/'model.tflite')
    if before!={str(p):file_sha256(p) for p in protected}:raise ValueError("Original artifacts changed during staging")
    print('Runtime assets:',folder,flush=True)
    print('Embedded default model:',args.default_variant,flush=True)
    print('Original models, dataset and normalization unchanged:',True,flush=True)
    print('Registry lists all three models; changing only model.tflite preserves registry contents.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,default=Path('training/dataset'))
    parser.add_argument('--stats',type=Path,default=DEFAULT_STATS)
    parser.add_argument('--models-dir',type=Path,default=Path('training/models'))
    parser.add_argument('--output-dir',type=Path,default=Path('deployment/runtime_assets'))
    parser.add_argument('--default-variant',default='M1_FP32',choices=['M1_FP32','M2_INT8','M3_STRUCTURED_INT8'])
    prepare(parser.parse_args())


if __name__=='__main__':main()
