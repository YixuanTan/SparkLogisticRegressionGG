package org.rpi.spark

// Import the needed libraries
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}


object gglr {
  // Transform each qualitative data in the data set into a double numeric value
  def main(args: Array[String]) {      
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // Read data into memory in - lazy loading
    val sc = new SparkContext("local[*]","LogisticRegression")
    //val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/*/{[5-9],1[0-5]}_[6-7]*_[2-4]*")
    val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/data1/{[5-9],1[0-5]}_[6-7]*_[2-4]*")
    println(data.count() + "\n") 
    
    
    // Prepare data for the logistic regression algorithm
    val parsedData = data.map{line => 
        val parts = line.split(",")
        LabeledPoint((parts(5)).toInt, Vectors.dense(parts.slice(0,5).map(x => x.toDouble).reverse))
    }
    
    /*
    val normalizer1 = new Normalizer()
    val data1 = parsedData.map(x => LabeledPoint(x.label.toInt, normalizer1.transform(x.features)))
    //data1.foreach(println)
    */
    
    // Creating a Scaler model that standardizes with both mean and SD
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedData.map(x => x.features))
    // Scale features using the scaler model
      val data1 = parsedData.map(x => LabeledPoint(x.label.toInt, scaler.transform(x.features)))
    //data1.foreach(println)

    
    //println(parsedData.take(3).mkString("\n"))
    
    // Split data into training (60%) and test (40%)
    val splits = parsedData.randomSplit(Array(0.9, 0.1), seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)
    
    
    // SVD processing
    /*
    val mat: RowMatrix = new RowMatrix(data1.map { line => line._2})
     
    val cov: Matrix = mat.computeCovariance()
    println("covariance matrix :")
    println(cov);
		*/
    
    // Compute the top 5 principal components.
    /*
    val pca = new PCA(5).fit(data1.map(_.features))
    
    // Project vectors to the linear space spanned by the top 5 principal
    // components, keeping the label
    val projected = data1.map(p => p.copy(features = pca.transform(p.features)))
    // $example off$
    val collect = projected.collect()
    println("Projected vector of principal component:")
    //collect.foreach { vector => println(vector) }
		*/
    
    
/*
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(6, computeU = true)
    val U: RowMatrix = svd.U // The U factor is a RowMatrix.
    val s: Vector = svd.s // The singular values are stored in a local dense vector.
    val V: Matrix = svd.V // The V factor is a local dense matrix.

    println("Left Singular vectors :")
    U.rows.foreach(println)
   
    println("Singular values are :")
    println(s)
   
    println("Right Singular vectors :")
    println(V)
*/
    
    
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
    
    // ROC Curve
    val roc = metrics.roc
    roc.map{ case (t, r) =>
      (s"FPR: $t, TPR: $r")
    }.coalesce(1,true).saveAsTextFile("ROC.txt")
    

    
    
    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
		
    
    sc.stop
  }

}