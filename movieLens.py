#!/usr/bin/python# -*- coding:utf-8 -*-import sys#from pyspark import SparkContextsys.path.append("/home/lei/spark-1.5.0-bin-hadoop2.3/python")try:    from pyspark import SparkContext    from pyspark import SparkConf    print("import success!")except ImportError as e:    print(e)sc=SparkContext("local")words=sc.parallelize(["a","b","c"])print words.count()