# Model Report: Scopus Subject Classifier

- Train size: 28855
- Test size: 7214
- Train accuracy: 1.0000
- Test accuracy: 0.9403
- Train-test gap: 0.0597
- 5-fold CV mean accuracy (on train set): 0.9352
- 5-fold CV std: 0.0016
- Majority-class baseline test accuracy: 0.1083

## Classification report (test set)

```
                                             precision    recall  f1-score   support

                    Artificial Intelligence       0.91      0.94      0.93       714
       Computational Theory AND Mathematics       0.96      0.97      0.97       781
Computer Graphics and Computer-Aided Design       0.96      0.97      0.97       774
       Computer Networks and Communications       0.93      0.90      0.92       723
              Computer Science Applications       0.91      0.84      0.87       697
    Computer Vision and Pattern Recognition       0.92      0.96      0.94       750
                          Computer software       0.97      0.99      0.98       764
                            Computer vision       0.93      0.91      0.92       622
                   General Computer Science       0.96      0.94      0.95       744
                 Human-Computer Interaction       0.93      0.96      0.95       645

                                   accuracy                           0.94      7214
                                  macro avg       0.94      0.94      0.94      7214
                               weighted avg       0.94      0.94      0.94      7214
```

## Confusion matrix (test set)

|                                             |   Artificial Intelligence |   Computational Theory AND Mathematics |   Computer Graphics and Computer-Aided Design |   Computer Networks and Communications |   Computer Science Applications |   Computer Vision and Pattern Recognition |   Computer software |   Computer vision |   General Computer Science |   Human-Computer Interaction |
|:--------------------------------------------|--------------------------:|---------------------------------------:|----------------------------------------------:|---------------------------------------:|--------------------------------:|------------------------------------------:|--------------------:|------------------:|---------------------------:|-----------------------------:|
| Artificial Intelligence                     |                       674 |                                      4 |                                             0 |                                      9 |                               9 |                                         0 |                   4 |                 5 |                          3 |                            6 |
| Computational Theory AND Mathematics        |                         1 |                                    758 |                                             1 |                                      1 |                               9 |                                         2 |                   2 |                 1 |                          5 |                            1 |
| Computer Graphics and Computer-Aided Design |                         2 |                                      1 |                                           752 |                                      2 |                               1 |                                         1 |                   3 |                 2 |                          4 |                            6 |
| Computer Networks and Communications        |                        17 |                                      1 |                                             3 |                                    653 |                              11 |                                        16 |                   5 |                 7 |                          2 |                            8 |
| Computer Science Applications               |                        23 |                                      9 |                                            14 |                                     18 |                             585 |                                         8 |                   6 |                 8 |                         11 |                           15 |
| Computer Vision and Pattern Recognition     |                         3 |                                      0 |                                             3 |                                      3 |                               2 |                                       718 |                   0 |                17 |                          1 |                            3 |
| Computer software                           |                         0 |                                      1 |                                             3 |                                      4 |                               3 |                                         0 |                 753 |                 0 |                          0 |                            0 |
| Computer vision                             |                        14 |                                      2 |                                             4 |                                      4 |                               5 |                                        19 |                   0 |               568 |                          1 |                            5 |
| General Computer Science                    |                         0 |                                     10 |                                             2 |                                      4 |                              14 |                                         6 |                   4 |                 3 |                        701 |                            0 |
| Human-Computer Interaction                  |                         5 |                                      0 |                                             0 |                                      4 |                               7 |                                         7 |                   0 |                 1 |                          0 |                          621 |
