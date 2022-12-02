# FILM: Frame Interpolation for Large Motion, in pytorch

Implementation of [FILM: Frame Interpolation for Large Motion](https://github.com/google-research/frame-interpolation) in Pytorch.

## Usage

- Clone this repository.
```bash
git clone https://github.com/yangxy/FILM-pytorch.git
cd FILM-pytorch
```

- Download our [pre-trained model](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/film.pth) and put them into ``ckpts/``.

- Test our models.
```bash
python demo.py
```

## Citation
If our work is useful for your research, please consider citing:

    @inproceedings{reda2022film,
	 title = {FILM: Frame Interpolation for Large Motion},
	 author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
	 booktitle = {European Conference on Computer Vision (ECCV)},
	 year = {2022}
	}
    