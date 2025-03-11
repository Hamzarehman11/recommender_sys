# Databricks notebook source
import mlflow

mlflow.pyspark.ml.autolog()

# COMMAND ----------

dbutils.fs.ls('FileStore/tables/')

# COMMAND ----------

ratings = spark.read.csv('/FileStore/tables/ratings.csv', header=True, inferSchema=True)

# COMMAND ----------

ratings.show()

# COMMAND ----------

movies = spark.read.csv('/FileStore/tables/movies.csv', header=True, inferSchema=True)

# COMMAND ----------

movies.show(truncate=False)

# COMMAND ----------

myRatings = spark.read.csv('/FileStore/tables/myratings_1_.csv', header=True, inferSchema=True)

# COMMAND ----------

myRatings.show(truncate=False)

# COMMAND ----------

myRatings = myRatings.dropna()

# COMMAND ----------

myRatings.show(truncate=False)

# COMMAND ----------

myRatings = myRatings.drop('title')
ratings = ratings.drop('timestamp')

# COMMAND ----------

myRatings.show(truncate=False)

# COMMAND ----------

ratings.show(truncate=False)

# COMMAND ----------

ratings.groupBy('userId').count().display()

# COMMAND ----------

spark.catalog.listTables()

# COMMAND ----------

ratings.createOrReplaceTempView("ratings")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT userId, Count(userId) AS Count
# MAGIC FROM ratings
# MAGIC GROUP BY userId

# COMMAND ----------

movies.show(truncate=False)

# COMMAND ----------

ratings.show(truncate=False)

# COMMAND ----------

movies.createOrReplaceTempView("movies")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT r.movieId, COUNT(r.movieId) as Count, m.title
# MAGIC FROM ratings r
# MAGIC JOIN movies m ON r.movieId = m.movieId
# MAGIC GROUP BY r.movieId, m.title
# MAGIC ORDER BY Count DESC
# MAGIC LIMIT 10

# COMMAND ----------

(training, test) = ratings.randomSplit([0.8,0.2], seed=100)
training = training.unionAll(myRatings)

# COMMAND ----------

from pyspark.ml.recommendation import ALS

als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating', seed=100)

model = als.fit(training)

# COMMAND ----------

predictions = model.transform(test).dropna()

predictions.show()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

rmse = evaluator.evaluate(predictions)
print(rmse)

# COMMAND ----------

sampleMoviesPrediction = model.transform(myRatings)
sampleMoviesPrediction.show()

# COMMAND ----------

rmseSampleMovies = evaluator.evaluate(sampleMoviesPrediction)

# COMMAND ----------

print(rmseSampleMovies)

# COMMAND ----------

from pyspark.sql import functions

myGeneratedMovies = movies.withColumn('userId', functions.expr("int('0')"))

# COMMAND ----------

myGeneratedMovies = model.transform(myGeneratedMovies)

# COMMAND ----------

myGeneratedMovies.show()

# COMMAND ----------

myGeneratedMovies = myGeneratedMovies.dropna()

# COMMAND ----------

myGeneratedMovies.show()

# COMMAND ----------

myGeneratedMovies.orderBy('prediction', ascending=False).show(truncate=False)

# COMMAND ----------

myGeneratedMovies.createOrReplaceTempView("myGeneratedMovies")

# COMMAND ----------

spark.catalog.listTables()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM myGeneratedMovies
# MAGIC ORDER BY prediction DESC
# MAGIC Limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM myGeneratedMovies
# MAGIC WHERE genres = 'Drama' OR genres = 'Comedy'
# MAGIC ORDER BY prediction DESC
# MAGIC Limit 10

# COMMAND ----------

userRec = model.recommendForAllUsers(10)

# COMMAND ----------

userRec.show(truncate=False)

# COMMAND ----------

from pyspark.sql.functions import explode

userRec.where(userRec.userId == 1).select('recommendations').withColumn('recommendations', explode('recommendations')).select('recommendations.movieId', 'recommendations.rating')\
    .join(movies,['movieId']).show(truncate=False)

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

parameters = ParamGridBuilder()\
    .addGrid(als.rank,[5,10,15])\
    .addGrid(als.regParam,[0.001, 0.005, 0.01, 0.05, 0.1])\
    .build()

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit

tvs = TrainValidationSplit()\
    .setSeed(100)\
    .setTrainRatio(0.75)\
    .setEstimatorParamMaps(parameters)\
    .setEstimator(als)\
    .setEvaluator(evaluator)


# COMMAND ----------

gridSearchModel = tvs.fit(training)

# COMMAND ----------

bestModel = gridSearchModel.bestModel

# COMMAND ----------

print('Best rank', bestModel.rank)
print('Best regParam', bestModel._java_obj.parent().getRegParam())

# COMMAND ----------

bestModelPrediction = bestModel.transform(test)

# COMMAND ----------

bestModelPrediction.show()

# COMMAND ----------

bestModelPrediction.printSchema()

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

bestModelPrediction.select("prediction").summary("count", "min", "max").show()


# COMMAND ----------

bestModelPrediction = bestModelPrediction.na.drop()

# COMMAND ----------

bestModelPrediction.select("prediction").summary("count", "min", "max").show()

# COMMAND ----------

print(type(evaluator))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    metricName="rmse", 
    labelCol="rating", 
    predictionCol="prediction"
)

# COMMAND ----------

print(type(evaluator))

# COMMAND ----------

evl = evaluator.evaluate(bestModelPrediction)
print(evl)

# COMMAND ----------


