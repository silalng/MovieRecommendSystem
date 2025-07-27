package com.invi00.content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

case class Movie(mid: Int, name: String, descri: String, timelong: String, issue: String,
                 shoot: String,language: String, genres: String, actors: String, directors: String )

case class MongoConfig(uri:String, db:String)

// 标准推荐对象，mid,score case
case class Recommendation(mid: Int, score:Double)

// 电影相似度（电影推荐）
case class MovieRecs(mid: Int, recs: Seq[Recommendation])

object ContentRecommender {
  // 定义常量
  val MONGODB_MOVIE_COLLECTION = "Movie"

  // 推荐表的名称
  val CONTENT_MOVIE_RECS = "ContentMovieRecs"

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://sparkRecommend:27017/recommender",
      "mongo.db" -> "recommender")

    // 创建spark session
    val sparkConf = new
        SparkConf().setMaster(config("spark.cores")).setAppName("StatisticsRecommender")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    implicit val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    import spark.implicits._

    //加载数据，并作预处理
    val movieTagsDF = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .map(
        //提取mid, name, genres三项作为原始内容特征，分词器默认按照空格做分词
        movie => (movie.mid, movie.name, movie.genres.map(genres => if (genres == '|') ' ' else genres))
      )
      .toDF("mid", "name", "genres")
      .cache()

    //核心代码:从内容信息中提取特征向量

    // 实例化一个分词器，默认按空格分
    val tokenizer = new Tokenizer().setInputCol("genres").setOutputCol("words")

    // 用分词器做转换
    val wordsData = tokenizer.transform(movieTagsDF)

    // 定义一个HashingTF工具
    val hashingTF = new
        HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    // 用 HashingTF 做处理
    val featurizedData = hashingTF.transform(wordsData)

    // 定义一个IDF工具
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    // 将词频数据传入，得到idf模型（统计文档）
    val idfModel = idf.fit(featurizedData)

    // 用tf-idf算法得到新的特征矩阵
    val rescaledData = idfModel.transform(featurizedData)

    // 从计算得到的 rescaledData 中提取特征向量
    val movieFeatures = rescaledData.map {
        case row => (row.getAs[Int]("mid"), row.getAs[SparseVector]("features").toArray)
      }
      .rdd
      .map(x => {
        (x._1, new DoubleMatrix(x._2))
      })

    // 计算笛卡尔积并过滤合并
    val contentMovieRecs = movieFeatures.cartesian(movieFeatures)
      .filter { case (a, b) => a._1 != b._1 }
      .map { case (a, b) =>
        val simScore = this.consinSim(a._2, b._2) // 求余弦相似度
        (a._1, (b._1, simScore))
      }.filter(_._2._2 > 0.6)
      .groupByKey()
      .map { case (mid, items) =>
        MovieRecs(mid, items.toList.map(x => Recommendation(x._1, x._2)))
      }.toDF()

    contentMovieRecs
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", CONTENT_MOVIE_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    //关闭spark
    spark.stop()
  }

  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }
}
