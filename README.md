# WhenDeepIsNotEnough
When Deep is not Enough: Towards Understanding Shallow and Continual Learning Models in Realistic Environmental Sound Classification for Robots

[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=python)](https://www.python.org/) [![avalanche-lib 0.2.1](https://img.shields.io/badge/Avalance--lib-0.2.1-blue)](https://github.com/ContinualAI/avalanche/tree/v0.2.1) [![tourch 1.13.0](https://img.shields.io/badge/tourch-1.13.0-blue)](https://pytorch.org/)


This repository provides the official implementation of the paper:
> **[When Deep is not Enough: Towards Understanding Shallow and Continual Learning Models in Realistic Environmental Sound Classification for Robots](https://lpaperlink) (International Journal of Humanoid Robotics)**<br>
>*‡[Omar Eldardeer](https://scholar.google.com/citations?user=2xry9p8AAAAJ&hl),  *‡[Francesco Rea](https://scholar.google.com/citations?user=6rh0-d8AAAAJ&hl),   *‡[Giulio Sandini](https://scholar.google.com/citations?user=5mSnPlwAAAAJ&hl),  and *‡[Doreen Jirak](https://scholar.google.com/citations?user=-HgMDDYAAAAJ&hl)<br>
> *Università di genova , ‡Istituto Italiano di Tecnologia,<br>
> Note: Research was conducted at Istituto Italiano di Tecnologia <br>
## Abstract

Although deep learning models are state-of-the-art models in audio classification, their fall short when applied in developmental robotic settings and human-robot interaction (HRI). The major drawback is that deep learning relies on supervised training with a large amount of data and annotations. In contrast, developmental learning strategies in human-robot interaction often deal with small-scale data acquired from HRI experiments and require the incremental addition of novel classes. Alternatively, shallow learning architectures that enable fast and yet robust learning are provided by simple distance metric-based learning and neural architectures implementing the reservoir computing paradigm. Similarly, continual learning algorithms receive more attention in the last years as they can integrate stable perceptual feature extraction using pre-trained deep learning models with open-set classification. As our research centers around reenacting the incremental learning of audio cues, we conducted a study on environmental sound classification using the iCaRL as well as the GDumb continual learning algorithms in comparison with a popular classifier in this domain, the knn classifier, as well as employing an Echo State Network. We contrast our results with those obtained
from a VGGish network that serves here as the performance upper bound that allows us to quantify the performance differences and to discuss current issues with continual learning in the audio domain. As only little is known about using shallow models or continual learning in the audio domain, we pass on additional techniques like data augmentation and create a simple experimental pipeline that is easy to reproduce. Although our selected algorithms are partially inferior in performance compared to the upper bound, our evaluation on three environmental sound datasets shows promising performance using continual learning for a subset of the dcase2019 challenge dataset and the ESC10 dataset. As we do not address benchmarking in this paper, our study provides a good foundation for further research and computational improvements on shallow and continual learning models for robotic applications in the audio domain.


## Datasets

* dcase_icub
* [ESC10](https://github.com/karolpiczak/ESC-50)
* [ESC50](https://github.com/karolpiczak/ESC-50)


## <a name="Citing SVOAWP"></a> Citation
For citing our paper please use the following BibTeX entry:
```BibTeX
@Article{Eldardeer2023,
author={Eldardeer, Omar
and Rea, Francesco
and Sandini, Giulio
and Jirak, Doreen},
title={When Deep is not Enough: Towards Understanding Shallow and Continual Learning Models in Realistic Environmental Sound Classification for Robots},
journal={International Journal of Humanoid Robotics},
year={2023},
month={Jun},
day={30},
issn={---},
doi={S0219843623500081},
url={https://doi.org/S0219843623500081}
}
```
---

## License

This work has been published under [CC-by-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.


## Contact Details


Omar Eldardeer
* [Twitter](https://twitter.com/omareldardear)
* [Google Scholar](https://scholar.google.com/citations?user=2xry9p8AAAAJ&hl)
* [Linkedin](https://www.linkedin.com/in/omar-eldardear/)
* [Website](https://www.iit.it/people/omar-eldardeer )


Note: The code this work is based on was developed during my PhD thesis (2022).
