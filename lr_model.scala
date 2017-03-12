package org.rpi.spark
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors, SparseVector}
import org.apache.log4j.Logger
import org.apache.log4j.Level
//import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import sys.process._
import org.apache.spark.sql.SparkSession;
import java.io._

object lr_model {
    def main(args: Array[String]) {  
    val spark = SparkSession
     .builder()
     .appName("SparkSessionZipsExample")
     .master("local[*]")
     .enableHiveSupport()
     .getOrCreate()

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    "rm -r /home/smartcoder/Documents/Developer/InerseGG/neuralNet/ACC/"!
    
    "rm -r /home/smartcoder/Documents/Developer/InerseGG/neuralNet/fmeasure/"!

    "rm -r /home/smartcoder/Documents/Developer/InerseGG/neuralNet/ROC/"!

    "rm -r /home/smartcoder/Documents/Developer/InerseGG/neuralNet/recall/"!

    "rm -r /home/smartcoder/Documents/Developer/InerseGG/neuralNet/res/"!

    // Read data into memory in - lazy loading
    val data = spark.read.format("libsvm").option("numFeatures", "5").load("/home/smartcoder/Documents/Developer/InerseGG/neuralNet/libsvm_formatted.txt").cache()

    val count = data.count()
    println(data.count() + "\n") 
    
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.8, 0.2))
    val imbalance = splits(0).cache()
    val test = splits(1).rdd.map(row => LabeledPoint(
       row.getAs[Double]("label").toInt,   
       SparseVector.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
    )).cache()
    
    //for(line <- imbalance) println(line.mkString("\n"))
    //test.show(1000)
    
    //down sampling to balance data
    val pos = imbalance.filter("label == 1.0").cache()
    val neg = imbalance.filter("label == 0.0").cache()
    println(pos)
    val numPos= pos.count()
    val trainingData = pos.union(neg.sample(false, 1.0 * numPos / (neg.count() - numPos))).rdd.map(row => LabeledPoint(
       row.getAs[Double]("label").toInt,   
       SparseVector.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))
    ))
    
    val testpos = test.filter(line => line.label == 1.0).cache()
    val testneg = test.filter(line => line.label == 0.0).cache()
    val testData = testpos.union(testneg.sample(false, 1.0 * testpos.count() / (testneg.count() - testpos.count())))
    
    // Train the model
    println("how many trainingData ? " + trainingData.count())
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
    // Here are the coefficient and intercept
    val weights: org.apache.spark.mllib.linalg.Vector = model.weights
    val intercept = model.intercept
    println(weights) 
    println(intercept) 
    //val weightsData: Array[Double] = weights.asInstanceOf[Vector].values

    //model.save(spark, "lr_model")

    
    println("how many testData ? " + testData.count())

    // Evaluate model on training examples and compute training error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count
    println("Training Error = " + trainErr)
    
    labelAndPreds.map { case (l, p) =>
      (s"$l, $p")
    }.coalesce(1,true).saveAsTextFile("/home/smartcoder/Documents/Developer/InerseGG/neuralNet/res")

    
    /*
    val res = new PrintWriter(new FileOutputStream(
        new File("/home/smartcoder/Documents/Developer/InerseGG/neuralNet/res.txt" ),
        true // append mode)
    ));

    labelAndPreds.foreach{
      line => res.write(line._1 + " " + line._2)
    }
    res.close
    */
    // add more metric output
    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold
    // Compute raw scores on the test set
    val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    
    // Instantiate metrics object
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    
    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.map { case (t, f) =>
      (s"$t, $f")
    }.coalesce(1,true).saveAsTextFile("/home/smartcoder/Documents/Developer/InerseGG/neuralNet/fmeasure")

		
    
    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.map{ case (t, p) =>
      (s"$t, $p")
    }.coalesce(1,true).saveAsTextFile("/home/smartcoder/Documents/Developer/InerseGG/neuralNet/ACC")
    
    val roc = metrics.roc
    roc.map{ case (t, r) =>
      (s"$t, $r")
    }.coalesce(1,true).saveAsTextFile("/home/smartcoder/Documents/Developer/InerseGG/neuralNet/ROC")
    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.map { case (t, r) =>
      (s"$t, $r")
    }.coalesce(1, true).saveAsTextFile("/home/smartcoder/Documents/Developer/InerseGG/neuralNet/recall")

  }
}