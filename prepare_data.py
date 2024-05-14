"""
Copyright: Wenyi Tang 2017-2019
Author: Wenyi Tang
Email: wenyi.tang@intel.com
Created Date: Dec 20th 2018

Prepare datasets and install VSR package for users.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

from tensorflow import keras

kutils = keras.utils

# For now VSR requires python>=3.5
if sys.version_info.major == 3 and sys.version_info.minor < 6:
  print("Python version is required >=3.6!")
  exit(-1)

# _DEFAULT_DATASET_PATH = '/mnt/data/datasets'
_DEFAULT_DATASET_PATH = '.vsr/datasets'
_DEFAULT_DOWNLOAD_DIR = '.vsr/downloads'
_DEFAULT_WEIGHTS_DIR = './Results'
# Contact me if any of these links un-accessed
DATASETS = {
  'DIV2K': {
    'DIV2K_train_HR.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
    'DIV2K_valid_HR.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
    'DIV2K_train_LR_unknown_X4.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip',
    'DIV2K_valid_LR_unknown_X4.zip': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip',
  },
  'SET5.zip': 'https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip',
  'SET14.zip': 'https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip',
  'SunHay80.zip': 'https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip',
  'Urban100.zip': 'https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip',
  'BSD300.tgz': 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz',
  'BSD500.tgz': 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz',
  '91image.rar': 'http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar',
  'waterloo.rar': 'http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar',
  # Google Drive File ID.
  # If you can't download from this file, visit url https://drive.google.com/open?id=<id>
  # paste the file id into position <id>.
  'GOPRO_Large.zip': '1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2', # https://drive.google.com/open?id=1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2
  'MCL-V.rar': '1z41hdqR-bqNLcUWllPePzkfQW-I_A9ny',
  'vid4.zip': '1ogEdifL_krqJnFAHfGNqOSMuUg_Ud6fb', # https://drive.google.com/open?id=1ogEdifL_krqJnFAHfGNqOSMuUg_Ud6fb
}
WEIGHTS = {
  'tensorflow': {
    'srcnn.tar.gz': 'https://github.com/LoSealL/Model/releases/download/srcnn/srcnn.tar.gz',
    'edsr.zip': 'https://github.com/LoSealL/Model/releases/download/edsr/edsr.zip',
    'dncnn.zip': 'https://github.com/LoSealL/Model/releases/download/DnCNN/dncnn.zip',
    'carn.zip': 'https://github.com/LoSealL/Model/releases/download/carn/carn.zip',
    'srdensenet.zip': '1aXAfRqZieY6mTfZUnErG84-9NfkQSeDw',
    'vdsr.zip': '1hW5YDxXpmjO2IfAy8f29O7yf1M3fPIg1',
    'msrn.zip': '1A0LoY3oB_VnArP3GzI1ILUNJbLAEjdtJ',
    'vespcn.zip': '19u4YpsyThxW5dv4fhpMj7c5gZeEDKthm',
    'dbpn.zip': '1ymtlOjhkGmad-od0zw7yTf17nWD4KMVi',
    'idn.zip': '1Fh3rtvrKKLAK27r518T1M_JET_LWZAFQ',
    'drsr_v2.zip': '1UrVNE6QMcQTW9Ks4P__JrRClb4IGTMYp',
    'drsr_sc2.zip': '1xIRVG7jbTM9fcLQkwyGyJIjwF2rTbNEJ',
    'drsr_sc4.zip': '1W-222rR2D2o-E99B4cXuUPBz2aCLuY_Z',
  },
  'pytorch': {
    'srcnn.zip': 'https://github.com/LoSealL/Model/releases/download/srcnn/Tsrcnn.zip',
    'espcn.zip': 'https://github.com/LoSealL/Model/releases/download/espcn/Tespcn.zip',
    'vdsr.zip': 'https://github.com/LoSealL/Model/releases/download/vdsr/Tvdsr.zip',
    'drcn.zip': 'https://github.com/LoSealL/Model/releases/download/drcn/Tdrcn.zip',
    'drrn.zip': 'https://github.com/LoSealL/Model/releases/download/drrn/Tdrrn.zip',
    'sofvsr.zip': 'https://github.com/LoSealL/Model/releases/download/sofvsr/SOFVSR_x4.zip',
    'carn.zip': 'https://github.com/LoSealL/Model/releases/download/carn/tcarn.zip',
    'edsr.pt': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'esrgan.zip': 'https://github.com/LoSealL/Model/releases/download/esrgan/esrgan.zip',
    'frvsr.zip': 'https://github.com/LoSealL/Model/releases/download/frvsr/FRVSR.zip',
    'mldn.zip': 'https://github.com/LoSealL/Model/releases/download/mldn/drn.zip',
    'crdn.zip': 'https://github.com/LoSealL/Model/releases/download/crdn/rsr.zip',
    'spmc.zip': 'https://github.com/LoSealL/Model/releases/download/spmc/spmc.zip',
    'rcan.zip': '10bEK-NxVtOS9-XSeyOZyaRmxUTX3iIRa',
    'rbpn.zip': '1Ozp5j-DBWJSpXY5GvxiEPKdfCaAbOXqu',
    'srmd.zip': '1ORKH05-aLSbQaWB4qQulIm2INoRufuD_',
    'dbpn.zip': '1PbhtuMz1zF3-d16dthurJ0xIQ9uyMvkz'
  }
}


def get_input(question):
  try:
    ans = input(question)
  except KeyboardInterrupt:
    ans = None
    print('\n', flush=True)  # user exit
    exit(0)
  return ans


def matches(str1, pattern):
  if not pattern:
    return str1
  ret = re.match(pattern.lower(), str1.lower())
  if ret:
    return str1


def user_input(name, defaults=False, pattern=None):
  if pattern.find('.*') < 0 and pattern.find('*') >= 0:
    pattern = pattern.replace('*', '.*')
  _name = matches(name, pattern)
  if _name is None:
    return
  question = 'Do you wish to download {}? '.format(_name)
  if defaults:
    return True
  else:
    question += '[y/N] '
  var = None
  while var is None:
    raw_ans = get_input(question)
    if raw_ans is None:
      print('\n', flush=True)  # user exit
      break
    elif raw_ans == '':
      var = defaults
      break
    ans = raw_ans.lower()
    if ans == 'y':
      var = True
    elif ans == 'n':
      var = False
    else:
      print('Invalid selection: {}'.format(raw_ans))
  return var


def download(name, url, path):
  fname = str(Path(path).resolve() / name)
  try:
    file = kutils.get_file(fname, url)
    return file
  except Exception:
    print('Unable to get file {}'.format(name))


def drive_download(name, url, path):
  file_path = Path(path) / name
  print(file_path)
  if file_path.exists():
    return file_path
  print(f"Google Drive Download API has been expired, "
        f"please download {name} from https://drive.google.com/open?id={url} directly."
        "\nFor more question, please message me at https://gitub.com/loseall.")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("filter", help="an re pattern to filter candidates.")
  parser.add_argument("--download_dir", type=str,
                      default=_DEFAULT_DOWNLOAD_DIR,
                      help="Specify download directory. "
                           "[{}]".format(_DEFAULT_DOWNLOAD_DIR))
  parser.add_argument("--data_dir", type=str,
                      default=_DEFAULT_DATASET_PATH,
                      help="Specify dataset extracted directory. "
                           "[{}]".format(_DEFAULT_DATASET_PATH))
  parser.add_argument("--weights_dir", type=str,
                      default=_DEFAULT_WEIGHTS_DIR,
                      help="Specify weights extracted directory. "
                           "[{}]".format(_DEFAULT_WEIGHTS_DIR))
  parser.add_argument("-q", "--quiet", action="store_true",
                      help="download quietly")
  args, _ = parser.parse_known_args()
  # make work dir
  Path(args.download_dir).mkdir(exist_ok=True, parents=True)

  def get_leaf(key: str, node: dict):
    for k, v in node.items():
      if isinstance(v, dict):
        for k2, v2 in get_leaf(k, v):
          yield Path(key) / k2, v2
      else:
        yield Path(key) / k, v

  need_to_download = {}
  try:
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    for k, v in get_leaf(args.data_dir, DATASETS):
      if user_input(k.stem, args.quiet, args.filter):
        need_to_download[k] = v
  except (FileNotFoundError, OSError):
    pass
  from VSR.Backend import BACKEND
  for k, v in get_leaf(args.weights_dir, WEIGHTS[BACKEND]):
    if user_input(k.stem, args.quiet, args.filter):
      need_to_download[k] = v
  need_to_extract = {}
  for k, v in need_to_download.items():
    if v[:4] == 'http':
      need_to_extract[k] = (k.parent,
                            download(k.name, v, args.download_dir))
    else:
      need_to_extract[k] = (k.parent,
                            drive_download(k.name, v, args.download_dir))
  for k, v in need_to_extract.values():
    if v is None:
      continue
    ext = Path(v).suffix
    if ext in ('.tar', '.tgz', '.gz', '.bz'):
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    elif ext in ('.zip',):
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile
    else:
      class copy:
        def __init__(self, src):
          self.src = src

        def __enter__(self):
          return self

        def __exit__(self, exc_type, exc_val, exc_tb):
          return

        def extractall(self, dst):
          import shutil
          shutil.copy(self.src, dst)

      is_match_fn = lambda x: True
      open_fn = copy
    if is_match_fn(v):
      with open_fn(v) as fd:
        try:
          fd.extractall(str(k.resolve()))
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          # TBD...
          pass
    else:
      print("[WARN] {} have to be uncompressed manually.".format(v))


if __name__ == '__main__':
  main()
