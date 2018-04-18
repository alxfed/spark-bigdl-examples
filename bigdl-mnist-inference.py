import argparse
import logging
import os
import glob

from bigdl.nn import layer
from bigdl.util import common
import numpy as np
import pyspark
import imageio



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
        '--input',
        help='input image path.'
    )
    parser.add_argument(
        '--executor-cores',
        '-c',
        default=4,
        type=int,
        help='Number of executor cores to use.'
    )
    parser.add_argument(
        '--model-dir',
        help='Trained model dir.'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    cores = args.executor_cores
    conf = (
        common.create_spark_conf()
            .setAppName('pyspark-mnist')
            .setMaster(args.master)
    )
    conf = conf.set('spark.executor.cores', cores)
    conf = conf.set('spark.cores.max', cores)
    conf.set("spark.jars",os.environ.get('BIGDL_JARS'))

    LOG.info('initialize with spark conf:')
    LOG.info(conf.getAll())
    sc = pyspark.SparkContext(conf=conf)
    common.init_engine()

    model = layer.Model.loadModel(
        args.model_dir + "/model.pb",
        args.model_dir + "/model.bin"
    )

    files = glob.glob(args.input+'/*.png')
    def mapper(x):
        image = imageio.imread('file://'+x).astype(np.float32).reshape(1, 28, 28)/255
        return image
    dataRDD = sc.parallelize(files).map(mapper)
    predictRDD  = dataRDD.map(lambda x: common.Sample.from_ndarray(x,np.array([2.0])))
    counts = model.predict(predictRDD).map(lambda x: (np.argmax(x)+1,1)).reduceByKey(lambda a,b: a+b)
    for x in counts.collect():
        LOG.info("%d count is %d",x[0],x[1])

    sc.stop()


if __name__ == '__main__':
    main()
