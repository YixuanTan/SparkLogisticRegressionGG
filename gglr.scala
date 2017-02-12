package org.rpi.spark

// Import the needed libraries
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import sys.process._


object gglr {
  // Transform each qualitative data in the data set into a double numeric value
  def main(args: Array[String]) {      
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    "rm -r /home/smartcoder/Documents/Developer/MSthesisData/ROC/"!
    // Read data into memory in - lazy loading
    val sc = new SparkContext("local[*]","LogisticRegression")
    //val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/*/{[5-9],1[0-5]}_[6-7]*_[2-4]*")
    //val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/data1/{[5-9],1[0-5]}_[6-7]*_[2-4]*")
    val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/combined.txt")
    val count = data.count()
    println(data.count() + "\n") 
       
    // Prepare data for the logistic regression algorithm
    //val parsedData = data.sample(false, 1.0*howManyTake/count).map{line => 
    val parsedData = data.map{line => val parts = line.split(",")
        LabeledPoint((parts(5)).toInt, Vectors.dense(parts.slice(0,5).map(x => x.toDouble).reverse))
    }
    
    // Split data into training (60%) and test (40%)
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)
    
    
    // Train the model
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
    
    // Evaluate model on training examples and compute training error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count
    println("Training Error = " + trainErr)
  
   
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
    
    /*
    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }
    
    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }
    
    // False positive rate by label
    //val fpr = metrics.FalsePositiveRate 
    //fpr.foreach { case (t, r) =>
      //println(s"Threshold: $t, FPR: $r")
    //}

        
    // Precision-Recall Curve
    val PRC = metrics.pr
    
    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }
    
    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }
    
    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)
    
    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)
    */
    // ROC Curve
    val roc = metrics.roc
    roc.map{ case (t, r) =>
      (s"FPR: $t, TPR: $r")
    }.coalesce(1,true).saveAsTextFile("/home/smartcoder/Documents/Developer/MSthesisData/ROC")
    

    
    
    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
		
    
    sc.stop
  }

}