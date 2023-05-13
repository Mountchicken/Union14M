union14m_benchmark_root = '../data/Union14M-L/Union14M-Benchmarks'

union14m_benchmark_artistic = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='artsitic/annotation.json',
    pipeline=None)

union14m_benchmark_contextless = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='contextless/annotation.json',
    pipeline=None)

union14m_benchmark_curve = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='curve/annotation.json',
    pipeline=None)

union14m_benchmark_general = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='general/annotation.json',
    pipeline=None)

union14m_benchmark_incomplete = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='incomplete/annotation.json',
    pipeline=None)

union14m_benchmark_incomplete_ori = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='incomplete_ori/annotation.json',
    pipeline=None)

union14m_benchmark_multi_oriented = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='multi_oriented/annotation.json',
    pipeline=None)

union14m_benchmark_multi_words = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='multi_words/annotation.json',
    pipeline=None)

union14m_benchmark_salient = dict(
    type='OCRDataset',
    data_root=union14m_benchmark_root,
    ann_file='salient/annotation.json',
    pipeline=None)
