## 原始baseline（alpha=1.0）

```shell
Saved Stage2 retrieval results to: webqsp_Feb13-13:49:52/retrieval_result_stage2_alpha1.0.pth
(srag1-2) ➜  retrieve git:(dev) ✗ python eval.py -d webqsp \
    -p webqsp_Feb13-13:49:52/retrieval_result_stage2_alpha1.0.pth \
    --k_list 50,100,200,400,500 | tee webqsp_Feb13-13:49:52/eval_stage2_alpha1.0.txt
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.946                        0.880              0.859
200       0.972                        0.928              0.909
400       0.985                        0.961              0.949
500       0.987                        0.969              0.961
```


## 锁Top50 替换71-100 MaxPool

```shell
(srag1-2) ➜  retrieve git:(dev) ✗ python eval.py -d webqsp \                                   
    -p webqsp_Feb13-13:49:52/retrieval_result_stage2_insert71-100.pth \
    --k_list 50,100,200,400,500
  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.905                        0.827              0.800
100       0.942                        0.875              0.859
200       0.973                        0.929              0.915
400       0.985                        0.962              0.949
500       0.987                        0.969              0.961
```
