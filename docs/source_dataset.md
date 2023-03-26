# Source Datasets of Union14M
- We collected labeled data from 14 publicly available datasets to construct **Union14M-L**. The details of these datasets are listed in the following table.

    |    Dataset    | Year  |                                               Link                                                | Lang.  |                                    License                                    |
    | :-----------: | :---: | :-----------------------------------------------------------------------------------------------: | :----: | :---------------------------------------------------------------------------: |
    |   KAIST[1]    | 2011  |          [link](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database)           | EN, KR |        [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)        |
    |   NEOCR[2]    | 2011  | [link](http://www.iapr-tc11.org/mediawiki/index.php?title=NEOCR:_Natural_Environment_OCR_Dataset) |   EN   |     [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)     |
    | Uber-Text[3]  | 2017  |                        [link](https://www.robots.ox.ac.uk/~vgg/data/text/)                        |   EN   |                                    Unknown                                    |
    |    RCTW[4]    | 2017  |                              [link](https://rctw.vlrlab.net/dataset)                              | EN, CH |                                    Unknown                                    |
    | IIIT-ILST[5]  | 2017  |             [link](http://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-ilst)              | EN, IN |           [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)           |
    |    MTWI[6]    | 2018  |       [link](https://tianchi.aliyun.com/competition/entrance/231685/information?lang=en-us)       | EN, CN |        [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)        |
    | COCOTextV2[7] | 2018  |                        [link](https://vision.cornell.edu/se3/coco-text-2/)                        |   EN   |           [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)           |
    |    LSVT[8]    | 2019  |                               [link](https://rrc.cvc.uab.es/?ch=16)                               | EN, CN |                                    Unknown                                    |
    |   MLT19[9]    | 2019  |                               [link](https://rrc.cvc.uab.es/?ch=15)                               | Multi  |        [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)        |
    |   ReCTS[10]   | 2019  |                               [link](https://rrc.cvc.uab.es/?ch=12)                               | EN, CN |                                    Unknown                                    |
    |    ArT[11]    | 2019  |                               [link](https://rrc.cvc.uab.es/?ch=14)                               | EN, CN |                                    Unknown                                    |
    | IntelOCR[12]  | 2021  |                   [link](https://github.com/cvdfoundation/open-images-dataset)                    |   EN   | [Apache License 2.0](https://github.com/openimages/dataset/blob/main/LICENSE) |
    |  TextOCR[13]  | 2021  |          [link](https://textvqa.org/textocr/dataset/https://textvqa.org/textocr/datase)           |   EN   |           [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)           |
    | HierText[14]  | 2022  |                   [link](https://github.com/google-research-datasets/hiertext)                    |   EN   |        [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)         |

- We collected unlabeled data from 3 publicly available datasets to construct **Union14M-U**. The details of these datasets are listed in the following table.

    |         Dataset         | Year  |                             Link                             | Lang. |                                    License                                    |
    | :---------------------: | :---: | :----------------------------------------------------------: | :---: | :---------------------------------------------------------------------------: |
    |       Book32[15]        | 2016  |      [link](https://github.com/uchidalab/book-dataset/)      |   -   |                                    Unknown                                    |
    | Conceptual Captions[16] | 2018  |   [link](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)    |   -   |                                     None                                      |
    |     OpenImages[17]      | 2020  | [link](https://github.com/cvdfoundation/open-images-dataset) |   -   | [Apache License 2.0](https://github.com/openimages/dataset/blob/main/LICENSE) |



- We are immensely grateful to the authors of the 17 datasets that we have consolidated into our work. If there is any problem about the license, please contact us.


## Dataset References
- [1] Jehyun Jung, SeongHun Lee, Min Su Cho, and Jin Hyung Kim. Touch TT: Scene text extractor using touchscreen in- terface. ETRI Journal, 33(1):78–88, 2011
- [2] Robert Nagy, Anders Dicker, and Klaus Meyer-Wegener. NEOCR: A configurable dataset for natural image text recognition. In International Workshop on Camera-Based Document Analysis and Recognition, pages 150–163. Springer, 2011.
- [3] Ying Zhang, Lionel Gueguen, Ilya Zharkov, Peter Zhang, Keith Seifert, and Ben Kadlec. Uber-text: A large-scale dataset for optical character recognition from street-level imagery. In SUNw: Scene Understanding Workshop-CVPR, volume 2017, page 5, 2017.
- [4] Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu, and Xiang Bai. ICDAR 2017 competition on reading chinese text in the wild (rctw-17). In ICDAR, volume 1, pages 1429–1434. IEEE, 2017.
- [5] Minesh Mathew, Mohit Jain, and CV Jawahar. Benchmarking scene text recognition in devanagari, telugu and malayalam. In ICDAR, volume 7, pages 42–46. IEEE, 2017.
- [6] Mengchao He, Yuliang Liu, Zhibo Yang, Sheng Zhang, Canjie Luo, Feiyu Gao, Qi Zheng, Yongpan Wang, Xin Zhang, and Lianwen Jin. ICPR 2018 contest on robust reading for multi-type web images. In ICPR, pages 7–12. IEEE, 2018.
- [7] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. COCO-Text: Dataset and benchmark for text detection and recognition in natural images. arXiv preprint arXiv:1601.07140, 2016.
- [8] Yipeng Sun, Jiaming Liu, Wei Liu, Junyu Han, Errui Ding, and Jingtuo Liu. Chinese Street View Text: Large-scale chinese text reading with partially supervised learning. In ICCV, pages 9086–9095, 2019.
- [9] Nibal Nayef, Yash Patel, Michal Busta, Pinaki Nath Chowdhury, Dimosthenis Karatzas, Wafa Khlif, Jiri Matas, Umapada Pal, Jean-Christophe Burie, Cheng-lin Liu, et al. ICDAR 2019 robust reading challenge on multi-lingual scene text detection and recognition—rrc-mlt-2019. In ICDAR, pages 1582–1587. IEEE, 2019.
- [10] Rui Zhang, Yongsheng Zhou, Qianyi Jiang, Qi Song, Nan Li, Kai Zhou, Lei Wang, Dong Wang, Minghui Liao, Mingkun Yang, et al. ICDAR 2019 robust reading challenge on reading chinese text on signboard. In ICDAR, pages 1577–1581. IEEE, 2019.
- [11] Chee Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. ICDAR 2019 robust reading
challenge on arbitrary-shaped text-rrc-art. In ICDAR, pages 1571–1576. IEEE, 2019.
- [12] Ilya Krylov, Sergei Nosov, and Vladislav Sovrasov. Openimages v5 text annotation and yet another mask text spotter. In ACML, pages 379–389. PMLR, 2021.
- [13] Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner. TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text. In CVPR, pages 8802–8812, 2021
- [14] Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Towards end-to-end unified scene text detection and layout analysis. In CVPR, pages 1049–1059, 2022.
- [15] Brian Kenji Iwana, Syed Tahseen Raza Rizvi, Sheraz Ahmed, Andreas Dengel, and Seiichi Uchida. Judging a book by its cover. arXiv preprint arXiv:1610.09204, 2016.
- [16] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual Captions: A cleaned, hypernymed, im- age alt-text dataset for automatic image captioning. In ACL, pages 2556–2565, 2018. 
- [17] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Ui- jlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4. Int. J. Comput. Vis., 128(7):1956–1981, 2020.