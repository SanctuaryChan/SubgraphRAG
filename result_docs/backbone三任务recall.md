# 原始baseline

## webqsp

```shell
(srag1-2) ➜  retrieve git:(backbone) ✗ python eval.py -d webqsp -p webqsp_Feb13-13:49:52/retrieval_result.pth  --k_list 50,100,200,400,500       
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.946                        0.880              0.859
200       0.972                        0.928              0.909
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
```


# backbone B0 网格

```shell
(srag1-2) ➜  retrieve git:(backbone) ✗ bash run_backbone_grid.sh webqsp_Feb13-13:49:52/cpt.pth webqsp
[Backbone grid] Running backbone_budget=20, output=webqsp_Feb13-13:49:52/retrieval_result_backbone_b20.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:00<00:00, 3089.93it/s]# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:53<00:00, 30.36it/s]Saved retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_backbone_b20.pth
Saved backbone metadata to: webqsp_Feb13-13:49:52/retrieval_result_backbone_b20.meta.json
Backbone stats: avg_backbone=19.97, avg_fill=79.81, backbone_empty=1, path_not_found=1
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.901                        0.827              0.804
100       0.946                        0.881              0.860
200       0.972                        0.928              0.909
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
[Backbone grid] Running backbone_budget=40, output=webqsp_Feb13-13:49:52/retrieval_result_backbone_b40.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:00<00:00, 3087.28it/s]# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:54<00:00, 29.99it/s]Saved retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_backbone_b40.pth
Saved backbone metadata to: webqsp_Feb13-13:49:52/retrieval_result_backbone_b40.meta.json
Backbone stats: avg_backbone=22.17, avg_fill=77.61, backbone_empty=1, path_not_found=1
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.899                        0.828              0.803
100       0.945                        0.882              0.861
200       0.972                        0.928              0.909
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
[Backbone grid] Running backbone_budget=60, output=webqsp_Feb13-13:49:52/retrieval_result_backbone_b60.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:00<00:00, 3110.39it/s]# skipped samples: 0
# relevant triples | median: 4 | mean: 20 | max: 699
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1639/1639 [00:54<00:00, 30.18it/s]Saved retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_backbone_b60.pth
Saved backbone metadata to: webqsp_Feb13-13:49:52/retrieval_result_backbone_b60.meta.json
Backbone stats: avg_backbone=22.17, avg_fill=77.61, backbone_empty=1, path_not_found=1
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.899                        0.828              0.803
100       0.945                        0.882              0.861
200       0.972                        0.928              0.909
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961

```