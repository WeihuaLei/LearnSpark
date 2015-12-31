#!/usr/bin/python
# -*-coding:utf-8 -*-


import sys
import numpy as np
from pprint import pprint
from featureproject import get_category_index, mappings

sys.path.append("/home/lei/spark-1.5.0-bin-hadoop2.3/python")

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SQLContext
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    from pyspark.mllib.tree import DecisionTree,RandomForest,GradientBoostedTrees

    from pyspark.ml.feature import StringIndexer,VectorIndexer
    from pyspark.ml import Pipeline
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier,GBTClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator


    print("import success!")
except ImportError as e:
    print e


def get_mapping(feat_rdds, cate_idx):
    """

    :param feat_rdds:
    :param cate_idx:
    :return:
    """
    mappings = {}
    for idx in cate_idx:
        from_0_idx= idx -1
        idx_map = feat_rdds.map(lambda entry: str(entry[from_0_idx])).distinct().collect()
        mappings[from_0_idx] = idx_map
    return mappings

def extract_features(entry,cateidx,numidx,cat_len,mappings):
    """

    :param entry:
    :param cateidx:
    :param numidx:
    :param cat_len:
    :param mappings:
    :return:
    """
    i = 0
    step = 0
    cat_vec = np.zeros(cat_len)
    for index in cateidx:
        field = str(entry[index])
        m = mappings[i]
        midx = m[field]
        cat_vec[midx+step]=1
        i = i + 1
        step = step + len(m)
    num_list = []
    for idx in numidx:
        num_list.append(entry[idx])
    num_vec = np.array(num_list)
    return np.concatenate((num_vec, cat_vec))


def testClassification(data):
    # Train a GradientBoostedTrees model.

    stringIndexer = StringIndexer(inputCol="label", outputCol="indexLabel")
    si_model = stringIndexer.fit(data)
    td = si_model.transform(data)

    rf = RandomForestClassifier(numTrees=5, maxDepth=4, labelCol="indexLabel",seed=13)

    trainData,testData = td.randomSplit([0.8,0.2],13)

    predictionDF = rf.fit(trainData).transform(testData)

    selected = predictionDF\
        .select('label','indexLabel','prediction','rawPrediction','probability')
    for row in selected.collect():
        print row

    scoresAndLabels = predictionDF\
       .map(lambda x: (float(x.probability.toArray()[1]), x.indexLabel))
    for sl in scoresAndLabels.collect():
        print sl
    evaluator = BinaryClassificationEvaluator(labelCol='indexLabel',metricName='areaUnderROC')
    metric = evaluator.evaluate(selected)
    print metric

def pipelineRF(dataDF):
    """

    :param train_data:
    :return:
    """

    print('pipeline starting...')
    labelIndexer_transModel = StringIndexer(inputCol='label',outputCol='indexLabel').fit(dataDF)
    featIndexer_transModel = VectorIndexer(inputCol="features", outputCol="indexed_features",maxCategories=37)\
                                    .fit(dataDF)

    #dtEstimator = DecisionTreeClassifier(featuresCol='indexed_features',labelCol='indexLabel',maxDepth=5,
    #                                      maxBins=40,minInstancesPerNode=1,minInfoGain=0.0,impurity='entropy')

    rfEstimator = RandomForestClassifier(labelCol='indexLabel',featuresCol='indexed_features',
                                         maxBins=40,seed=13)

    pipeline = Pipeline(stages=[labelIndexer_transModel,featIndexer_transModel,rfEstimator])

    paramGrid = ParamGridBuilder()\
        .addGrid(rfEstimator.maxDepth,[5,10,30])\
        .addGrid(rfEstimator.numTrees,[20,50,100]).build()

    evaluator =BinaryClassificationEvaluator(labelCol='indexLabel',
                                             rawPredictionCol='rawPrediction',
                                             metricName='areaUnderROC')
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=10)

    cvModel = cv.fit(dataDF)
    print("pipeline end..., cvModel  was fit using parameters:\n")
    pprint(cvModel.explainParams())


    predictionDF = cvModel.transform(dataDF)

    selected = predictionDF\
        .select('label','indexLabel','prediction','rawPrediction','probability')
    for row in selected.take(5):
        print row

    aucMetric = evaluator.evaluate(selected)
    print("auc of test data is:%.3f" % aucMetric)


def noneBalanceSampling(data):
    """

    :param data:
    :return:
    """
    underSamples = data.filter(data.label==1.0).sample(False,0.1,seed=13)
    overSamples = data.filter(data.label==0.0).sample(True,1.1,seed=13)
    samplesAll=underSamples.unionAll(overSamples)
    return samplesAll

def costSensitiveLearn(confuseMatrix):
    """

    :param confuseMatrix:
    :return:
    """


def loadLabeledPoint(sc,trainsXY):

    entries = trainsXY.map(lambda line: [float(ln.strip('"')) for ln in line.split(",")])
    features_label = entries.map(lambda entry: LabeledPoint(entry[-1], entry[:-1]))
    return features_label


if __name__ == "__main__":
    sc = SparkContext("local", "Classify")
    sqlContext = SQLContext(sc)
    trains_xy = sc.textFile("/home/lei/data/credit_prediction/train_xy.csv")
    featsLabels = loadLabeledPoint(sc,trains_xy).sample(False,0.05,13)


    #*********** pyspark.ML Pipeline method ***********
    dataDF = sqlContext.createDataFrame(featsLabels)


    #sampled = noneBalanceSampling(people_credit)
    testClassification(dataDF)
    pipelineRF(dataDF)



