---
license: apache-2.0
---
# UMT-msrvtt-7k Model Card
UMT-msrvtt-7k is the model used to compute UMTScore for the [FETV](https://github.com/llyx97/FETV) benchmark. It is initialized from the [UMT model](https://github.com/OpenGVLab/unmasked_teacher/blob/main/multi_modality/MODEL_ZOO.md) (UMT-L/16, 25M) and is fined-tuned on the 7k training split of MSR-VTT for video-text retrieval.

# Citation
```bibtex
@article{liu2023fetv,
  title   = {FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation},
  author  = {Yuanxin Liu and Lei Li and Shuhuai Ren and Rundong Gao and Shicheng Li and Sishuo Chen and Xu Sun and Lu Hou},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2311.01813}
}
```
```
@article{li2023unmasked,
      title={Unmasked Teacher: Towards Training-Efficient Video Foundation Models}, 
      author={Kunchang Li and Yali Wang and Yizhuo Li and Yi Wang and Yinan He and Limin Wang and Yu Qiao},
      year = {2023}
      journal = {arXiv preprint arXiv: 2303.16058}
}
```