import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.classification.LogisticRegression;

import static org.apache.spark.sql.functions.*;

public class SentimentAnalysis {
    public static void main(String[] args) {

        // 0️. Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("SentimentAnalysis")
                .config("spark.sql.warehouse.dir", "/user/hive/warehouse")
                .config("hive.metastore.uris", "thrift://localhost:9083")
                .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/sentimentdb")
                .enableHiveSupport()
                .getOrCreate();

        // 1️. Load from Hive
        Dataset<Row> df = spark.sql(
                "SELECT brand, primaryCategories, reviews_rating, reviews_text, reviews_doRecommend " +
                "FROM sentimentdb.reviews " +
                "WHERE reviews_text IS NOT NULL"
        );

        // 2️. Add label (positive if rating >= 3)
        Dataset<Row> labeled = df.withColumn("label",
                when(col("reviews_rating").cast("int").geq(3), 1).otherwise(0));

        // 3️. Text processing pipeline
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("reviews_text")
                .setOutputCol("words");

        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered");

        HashingTF hashingTF = new HashingTF()
                .setInputCol("filtered")
                .setOutputCol("features");

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10);

        Pipeline pipeline = new Pipeline()
                .setStages(new org.apache.spark.ml.PipelineStage[]{tokenizer, remover, hashingTF, lr});

        // 4️. Train model
        PipelineModel model = pipeline.fit(labeled);

        // 5️. Predict sentiment
        Dataset<Row> predictions = model.transform(labeled)
                .select("brand", "primaryCategories", "reviews_text", "reviews_rating", "reviews_doRecommend", "prediction");

        predictions.createOrReplaceTempView("sentiment_results");

        // 6️. Write detailed results to MongoDB (results collection)
        predictions.write()
                .format("mongo")
                .option("uri", "mongodb://127.0.0.1:27017/sentimentdb.results")
                .mode("overwrite")
                .save();

        // 7️. Aggregate trends per category and brand combinations
        Dataset<Row> trendSummary = spark.sql(
                "SELECT " +
                        "primaryCategories AS category, " +
                        "brand, " +
                        "COUNT(*) AS total_reviews, " +
                        "SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS positive_reviews, " +
                        "SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS negative_reviews, " +
                        "ROUND(SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS positive_percentage, " +
                        "ROUND(SUM(CASE WHEN reviews_doRecommend = 'TRUE' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS recommend_percentage " +
                        "FROM sentiment_results " +
                        "GROUP BY primaryCategories, brand " +
                        "ORDER BY positive_percentage DESC"
        );

        // 8️. Write trend summary to MongoDB (trend_summary collection)
        trendSummary.write()
                .format("mongo")
                .option("uri", "mongodb://127.0.0.1:27017/sentimentdb.trend_summary")
                .mode("overwrite")
                .save();

        // 9️. Optionally save trend summary also in Hive (optional)
        trendSummary.write()
                .mode("overwrite")
                .saveAsTable("sentimentdb.trend_summary");

        //  10. Stop Spark session
        spark.stop();
    }
}
