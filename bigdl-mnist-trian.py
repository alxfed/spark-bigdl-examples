import argparse
import logging
import os
import glob

from bigdl.nn import criterion
from bigdl.nn import layer
from bigdl.optim import optimizer
from bigdl.util import common
import pyspark
import imageio
import numpy as np
from mlboardclient.api import client

logging.basicConfig(
    format='%(asctime)s %(levelname)-10s %(name)-25s [-] %(message)s',
    level='INFO'
)
logging.root.setLevel(logging.INFO)
LOG = logging.getLogger('train')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--master',
        default=os.environ.get('SPARK_MASTER'),
        help='URI to spark master.'
    )
    parser.add_argument(
        '--batch-size',
        default=20,
        type=int,
        help='Batch size (number of simultaneously processed images).'
    )
    parser.add_argument(
        '--executor-cores',
        '-c',
        default=4,
        type=int,
        help='Number of executor cores to use.'
    )
    parser.add_argument(
        '--output-dir',
        default=os.environ.get('TRAINING_DIR')+'/'+os.environ.get('BUILD_ID'),
        help='Trained model output dir.'
    )
    parser.add_argument(
        '--data-dir',
        default=os.environ.get('DATA_DIR'),
        help='Trained model output dir.'
    )
    parser.add_argument(
        '--epoch',
        default=1,
        type=int,
        help='How many epoch to train.'
    )
    return parser


def build_model(class_num):
    model = layer.Sequential()
    model.add(layer.Reshape([1, 28, 28]))
    model.add(layer.SpatialConvolution(1, 6, 5, 5))
    model.add(layer.Tanh())
    model.add(layer.SpatialMaxPooling(2, 2, 2, 2))
    model.add(layer.Tanh())
    model.add(layer.SpatialConvolution(6, 12, 5, 5))
    model.add(layer.SpatialMaxPooling(2, 2, 2, 2))
    model.add(layer.Reshape([12 * 4 * 4]))
    model.add(layer.Linear(12 * 4 * 4, 100))
    model.add(layer.Tanh())
    model.add(layer.Linear(100, class_num))
    model.add(layer.LogSoftMax())
    return model

def main():
    parser = get_parser()
    args = parser.parse_args()

    # BATCH_SIZE must be multiple of <executor.cores>:
    # in this case multiple of 3: 3,6,9,12 etc.
    if args.batch_size % args.executor_cores != 0:
        raise RuntimeError(
            'batch size must be multiple of <executor-cores> parameter!'
        )

    cores = args.executor_cores
    batch_size = args.batch_size
    conf = (
        common.create_spark_conf()
            .setAppName('pyspark-mnist')
            .setMaster(args.master)
    )
    conf = conf.set('spark.executor.cores', cores)
    conf = conf.set('spark.cores.max', cores)
    conf.set("spark.jars",os.environ.get('BIGDL_JARS'))

    LOG.info('initialize with spark conf:')
    sc = pyspark.SparkContext(conf=conf)
    common.init_engine()

    LOG.info('initialize training RDD:')

    ##Files from kuberlab dataset
    files = glob.glob(os.environ.get('DATA_DIR')+'/train/*.png')
    LOG.info('Train size: %d',len(files))
    def mapper(x):
        label = int(x.split('/')[-1].split('-')[-1][:-4])+1
        image = imageio.imread('file://'+x).astype(np.float32).reshape(1,28,28)/255
        return common.Sample.from_ndarray(image, label)
    train_rdd = sc.parallelize(files).map(mapper)

    opt = optimizer.Optimizer(
        model=build_model(10),
        training_rdd=train_rdd,
        criterion=criterion.ClassNLLCriterion(),
        optim_method=optimizer.SGD(
            learningrate=0.01, learningrate_decay=0.0002
        ),
        end_trigger=optimizer.MaxEpoch(args.epoch),
        batch_size=batch_size
    )
    trained_model = opt.optimize()
    LOG.info("training finished")
    LOG.info('saving model...')
    path = args.output_dir
    if not os.path.exists(path):
        os.makedirs(path)
    trained_model.saveModel(
        path + '/model.pb',
        path + '/model.bin',
        over_write=True
    )
    client.update_task_info({'checkpoint_path': path,'model_path': path})
    LOG.info('successfully saved!')
    files = glob.glob(os.environ.get('DATA_DIR')+'/test/*.png')
    LOG.info('Validation size: %d',len(files))
    test_rdd = sc.parallelize(files).map(mapper)
    results = trained_model.evaluate(test_rdd, batch_size , [optimizer.Top1Accuracy()])
    accuracy = results[0].result
    client.update_task_info({'test_accuracy': float(accuracy)})
    sc.stop()


if __name__ == '__main__':
    main()
