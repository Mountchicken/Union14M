# Code Migration

MMOCR has been designed in a way that there are a lot of shortcomings in the initial version in order to balance the tasks of text detection, recognition and key information extraction. In this 1.0 release, MMOCR synchronizes its new model architecture to align as much as possible with the overall OpenMMLab design and to achieve structural uniformity within the algorithm library. Although this upgrade is not fully backward compatible, we summarize the changes that may be of interest to developers for those who need them.

## Fundamental Changes

Functional boundaries of modules has not been clearly defined in MMOCR 0.x. In MMOCR 1.0, we address this issue by refactoring the design of model modules. Here are some major changes in 1.0:

- MMOCR 1.0 no longer supports named entity recognition tasks since it's not in the scope of OCR.

- The module that computes the loss in a model is named as *Module Loss*, which is also responsible for the conversion of gold annotations into loss targets. Another module, *Postprocessor*, is responsible for decoding the model raw output into `DataSample` for the corresponding task at prediction time.

- The inputs of all models are now organized as a dictionary that consists of two keys: `inputs`, containing the original features of the images, and `List[DataSample]`, containing the meta-information of the images. At training time, the output format of a model is standardized to a dictionary containing the loss tensors. Similarly, a model generates a sequence of `DataSample`s containing the prediction outputs in testing.

- In MMOCR 0.x, the majority of classes named `XXLoss` have the implementations closely bound to the corresponding model, while their names made users hard to tell them apart from other generic losses like `DiceLoss`. In 1.0, they are renamed to the form `XXModuleLoss`. (e.g. `DBLoss` was renamed to `DBModuleLoss`). The key to their configurations in config files is also changed from `loss` to `module_loss`.

- The names of generic loss classes that are not related to the model implementation are kept as `XXLoss`. (e.g. [`MaskedBCELoss`](mmocr.models.common.losses.MaskedBCELoss)) They are all placed under `mmocr/models/common/losses`.

- Changes under `mmocr/models/common/losses`: `DiceLoss` is renamed to [`MaskedDiceLoss`](mmocr.models.common.losses.MaskedDiceLoss). `FocalLoss` has been removed.

- MMOCR 1.0 adds a *Dictionary* module which originates from *label converter*. It is used in text recognition and key information extraction tasks.

## Text Detection Models

### Key Changes (TL;DR)

- The model weights from MMOCR 0.x still works in the 1.0, but the fields starting with `bbox_head` in the state dict `state_dict` need to be renamed to `det_head`.

- `XXTargets` transforms, which were responsible for genearting detection targets, have been merged into `XXModuleLoss`.

### SingleStageTextDetector

- The original inheritance chain was `mmdet.BaseDetector->SingleStageDetector->SingleStageTextDetector`. Now `SingleStageTextDetector` is directly inherited from `BaseDetector` without extra dependency on MMDetection, and `SingleStageDetector` is deleted.

- `bbox_head` is renamed to `det_head`.

- `train_cfg`, `test_cfg` and `pretrained` fields are removed.

- `forward_train()` and `simple_test()` are refactored to `loss()` and `predict()`. The part of `simple_test()` that was responsible for splitting the raw output of the model and feeding it into `head.get_bounary()` is integrated into `BaseTextDetPostProcessor`.

- `TextDetectorMixin` has been removed since its implementation overlaps with `TextDetLocalVisualizer`.

### Head

- `HeadMixin`, the base class that `XXXHead` had to inherit from in version 0.x, has been replaced by `BaseTextDetHead`. `get_boundary()` and `resize_boundary()` are now rewritten as `__call__()` and `rescale()`  in `BaseTextDetPostProcessor`.

### ModuleLoss

- Data transforms `XXXTargets` in text detection tasks are all moved to `XXXModuleLoss._get_target_single()`. Target-related configurations are no longer specified in the data pipeline but in `XXXLoss` instead.

### Postprocessor

- The logic in the original `XXXPostprocessor.__call__()` are transferred to the refactored `XXXPostprocessor.get_text_instances()`.

- `BasePostprocessor` is refactored to `BaseTextDetPostProcessor`. This base class splits and processes the model output predictions one by one and supports automatic scaling of the output polygon or bounding box based on `scale_factor`.

## Text Recognition

### Key Changes (TL;DR)

- Due to the change of the character order and some bugs in the model architecture being fixed, the recognition model weights in 0.x can no longer be directly used in 1.0. We will provide a migration script and tutorial for those who need it.

- The support of SegOCR has been removed. TPS-CRNN will still be supported in a later version.

- Test time augmentation will be supported in the upcoming release.

- *Label converter* module has been removed and its functions have been split into *Dictionary*, *ModuleLoss* and *Postprocessor*.

- The definition of `max_seq_len` has been unified and now it represents the original output length of the model.

### Label Converter

- The original label converters had spelling errors (written as label convertors). We fixed them by removing label converters from this project.

- The part responsible for converting characters/strings to and from numeric indexes was extracted to *Dictionary*.

- In older versions, different label converters would have different special character sets and character order. In version 0.x, the character order was as follows.

| Converter                       | Character order                           |
| ------------------------------- | ----------------------------------------- |
| `AttnConvertor`, `ABIConvertor` | `<UKN>`, `<BOS/EOS>`, `<PAD>`, characters |
| `CTCConvertor`                  | `<BLK>`, `<UKN>`, characters              |

In 1.0, instead of designing different dictionaries and character orders for different tasks, we have a unified *Dictionary* implementation with the character order always as characters, \<BOS/EOS>, \<PAD>, \<UKN>. \<BLK> in `CTCConvertor` has been equivalently replaced by \<PAD>.

- *Label convertor* originally supported three ways to initialize dictionaries: `dict_type`, `dict_file` and `dict_list`, which are now reduced to `dict_file` only in `Dictionary`. Also, we have put those pre-defined character sets originally supported in `dict_type` into `dicts/` directory now. The corresponding mapping is as follows:

  | MMOCR 0.x: `dict_type` | MMOCR 1.0: Dict path                   |
  | ---------------------- | -------------------------------------- |
  | DICT90                 | dicts/english_digits_symbols.txt       |
  | DICT91                 | dicts/english_digits_symbols_space.txt |
  | DICT36                 | dicts/lower_english_digits.txt         |
  | DICT37                 | dicts/lower_english_digits_space.txt   |

- The implementation of `str2tensor()` in *label converter* has been moved to `ModuleLoss.get_targets()`. The following table shows the correspondence between the old and new method implementations. Note that the old and new implementations are not identical.

  | MMOCR 0.x                                                 | MMOCR 1.0                               | Note                                                                                                     |
  | --------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------- |
  | `ABIConvertor.str2tensor()`, `AttnConvertor.str2tensor()` | `BaseTextRecogModuleLoss.get_targets()` | The different implementations between `ABIConvertor.str2tensor()` and `AttnConvertor.str2tensor()` have been unified in the new version. |
  | `CTCConvertor.str2tensor()`                               | `CTCModuleLoss.get_targets()`           |                                                                                                          |

- The implementation of `tensor2idx()` in *label converter* has been moved to `Postprocessor.get_single_prediction()`. The following table shows the correspondence between the old and new method implementations. Note that the old and new implementations are not identical.

  | MMOCR 0.x                                                 | MMOCR 1.0                                        |
  | --------------------------------------------------------- | ------------------------------------------------ |
  | `ABIConvertor.tensor2idx()`, `AttnConvertor.tensor2idx()` | `AttentionPostprocessor.get_single_prediction()` |
  | `CTCConvertor.tensor2idx()`                               | `CTCPostProcessor.get_single_prediction()`       |

## Key Information Extraction

### Key Changes (TL;DR)

- Due to changes in the inputs to the model, the model weights obtained in 0.x can no longer be directly used in 1.0.

### KIEDataset & OpensetKIEDataset

- The part that reads data is kept in `WildReceiptDataset`.

- The part that additionally processes the nodes and edges is moved to `LoadKIEAnnotation`.

- The part that uses dictionaries to transform text is moved to `SDMGRHead.convert_text()`, with the help of *Dictionary*.

- The part of `compute_relation()` that computes the relationships between text boxes is moved to `SDMGRHead.compute_relations()`. It's now done inside the model.

- The part that evaluates the model performance is done in [`F1Metric`](mmocr.evaluation.metric.F1Metric).

- The part of `OpensetKIEDataset` that processes model's edge outputs is moved to `SDMGRPostProcessor`.

### SDMGR

- `show_result()` is integrated into `KIEVisualizer`.

- The part of `forward_test()` that post-processes the output is organized in `SDMGRPostProcessor`.

## Utils Migration

Utility functions are now grouped together under `mmocr/utils/`. Here are the scopes of the files in this directory:

- bbox_utils.py: bounding box related functions.
- check_argument.py: used to check argument type.
- collect_env.py: used to collect running environment.
- data_converter_utils.py: used for data format conversion.
- fileio.py: file input and output related functions.
- img_utils.py: image processing related functions.
- mask_utils.py: mask related functions.
- ocr.py: used for MMOCR inference.
- parsers.py: used for parsing datasets.
- polygon_utils.py: polygon related functions.
- setup_env.py: used for initialize MMOCR.
- string_utils.py: string related functions.
- typing.py: defines the abbreviation of types used in MMOCR.
