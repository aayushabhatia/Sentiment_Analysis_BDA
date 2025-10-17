import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, HashingTF, IDF}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline

object SentimentAnalysis {
  def main(args: Array[String]): Unit = {

    // 0️. Initialize Spark Session
    val spark = SparkSession.builder()
      .appName("SentimentAnalysis")
      .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
      .config("hive.metastore.uris", "thrift://localhost:9083")
      .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/sentimentdb") // base URI
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._

    // 1️. Load from Hive
    val df = spark.sql(
      """
        SELECT brand, primaryCategories, reviews_rating, reviews_text, reviews_doRecommend
        FROM sentimentdb.reviews
        WHERE reviews_text IS NOT NULL
      """
    )

    // 2️. Add label (positive if rating >= 3)
    val labeled = df.withColumn("label", when($"reviews_rating" >= 3, 1).otherwise(0))

    // 3️. Text processing pipeline
    val tokenizer = new Tokenizer().setInputCol("reviews_text").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("features")
    val lr = new LogisticRegression().setMaxIter(10)

    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, lr))

    // 4️. Train sentiment model
    val model = pipeline.fit(labeled)

    // 5️. Predict sentiment
    val predictions = model.transform(labeled)
      .select("brand", "primaryCategories", "reviews_text", "reviews_rating", "reviews_doRecommend", "prediction")

    // 6️. Write detailed results to MongoDB (`results` collection)
    predictions.write
      .format("mongo")
      .option("uri", "mongodb://127.0.0.1:27017/sentimentdb.results")
      .mode("overwrite")
      .save()

    // 7️. Aggregate trends per category and brand combinations
    predictions.createOrReplaceTempView("sentiment_results")

    val trendSummary = spark.sql(
      """
        SELECT
          primaryCategories AS category,
          brand,
          COUNT(*) AS total_reviews,
          SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS positive_reviews,
          SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS negative_reviews,
          ROUND(SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS positive_percentage,
          ROUND(SUM(CASE WHEN reviews_doRecommend = 'TRUE' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS recommend_percentage
        FROM sentiment_results
        GROUP BY primaryCategories, brand
        ORDER BY positive_percentage DESC
      """
    )

    // 8️. Write trend summary to MongoDB (`trend_summary` collection)
    trendSummary.write
      .format("mongo")
      .option("uri", "mongodb://127.0.0.1:27017/sentimentdb.trend_summary")
      .mode("overwrite")
      .save()

    // 9️. Stop Spark session
    spark.stop()
  }
}