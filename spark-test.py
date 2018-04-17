import pyspark
from pyspark import SparkConf, SparkContext
import os
conf = SparkConf().setAppName('test').setMaster(os.environ.get("SPARK_MASTER"))
sc = SparkContext(conf=conf)
data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)
v = distData.reduce(lambda a, b: a + b)
print(v)
sc.stop()