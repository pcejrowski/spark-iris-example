package pl.pcejrowski

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object IrisClustering {

  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf()
      .setAppName("user-topics-clustering")
      .setMaster("local[4]")
    val sc: SparkContext = new SparkContext(conf)

    val sourceCSV: RDD[Array[String]] = sc.textFile("src/main/resources/iris.csv")
      .repartition(4)
      .map(_.split(","))

    val labels: RDD[String] = sourceCSV
      .map(_.drop(4).head)

    val data: RDD[Vector] = sourceCSV
      .map(_.take(4))
      .map(x => Vectors.dense(x.map(_.toDouble)))

    val kMeans: KMeansModel = new KMeans()
      .setK(3)
      .setMaxIterations(100)
      .run(data)

    kMeans
      .predict(data)
      .zip(labels)
      .map {
        case (received, real) => s"$real,$received"
      }
      .repartition(1)
      .saveAsTextFile("results")
  }
}