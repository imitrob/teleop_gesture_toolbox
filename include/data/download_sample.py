
from pathlib import Path
import gdown
import tarfile

if __name__ == '__main__':
    path = Path(__file__)
    
    # 1    
    url_model = "https://drive.google.com/uc?id=1jyDatUJy10sdXmLjPEdHL1cfSo-RZ4ct"
    out_model = path.parent.joinpath('trained_networks', 'PyMC3-main-set-3.pkl')
    
    input(f"Download sample network to {out_model}")
    gdown.download(url_model, str(out_model), quiet=False)

    # 2
    out_learn = path.parent.joinpath('learning.tar.xz')
    url_learn = "https://drive.google.com/uc?id=1Jitk-MxzczreZ81PuO86xTapuSkBMOb-"
    
    input(f"Download sample network to {out_learn}")
    gdown.download(url_learn, str(out_learn), quiet=False)
    
    with tarfile.open(out_learn) as f:
        f.extractall(str(out_learn.parent))