package org.maj.sm.ml;

import com.google.common.collect.Maps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;

import java.util.Map;

/**
 * @author shamik.majumdar
 */
public class AppRunner {

    final static Logger logger = LoggerFactory.getLogger(AppRunner.class);
    private final DataService dataService = new DataService();
    private final ClassificationService classificationService = new ClassificationService();
    private final AccuracyService accuracyService = new AccuracyService();

    public static void main(String... args) throws Exception {

        AppRunner appRunner = new AppRunner();

        Instances data = appRunner.dataService.getData("/data.txt");
        data.setClassIndex(data.numAttributes() - 1);
        // Do 10-split cross validation
        Instances[][] split = appRunner.dataService.crossValidationSplit(data,10);

        // Separate split into training and testing arrays
        Instances[] trainingSplits = split[0];
        Instances[] testingSplits = split[1];

        // Use a set of classifiers
        Classifier[] models = {
                new J48(), // a decision tree
                new PART(),
                new DecisionTable(),//decision table majority classifier
                new DecisionStump() //one-level decision tree
        };


        Map<String,FastVector> predictionMap = Maps.newHashMap();

        // Run for each model
        for (int j = 0; j < models.length; j++) {

            // Collect every group of predictions for current model in a FastVector
            FastVector predictions = new FastVector();

            // For each training-testing split pair, train and test the classifier
            for (int i = 0; i < trainingSplits.length; i++) {
                Evaluation validation = appRunner.classificationService.classify(models[j], trainingSplits[i], testingSplits[i]);

                predictions.appendElements(validation.predictions());
                logger.debug(models[j].toString());
            }

            predictionMap.put(models[j].getClass().getSimpleName(),predictions);

            // Calculate overall accuracy of current classifier on all splits
            double accuracy = appRunner.accuracyService.calculateAccuracy(predictions);

            // Print current classifier's name and accuracy in a complicated,
            // but nice-looking way.
            logger.info("Accuracy of " + models[j].getClass().getSimpleName() + ": "
                    + String.format("%.2f%%", accuracy)
                    + "\n---------------------------------");
        }


        String[] allModels = predictionMap.keySet().toArray(new String[]{});

        if (allModels.length >= 2){
            StatisticalSignificance significance = new McNemarTest();
            for (int i=0; i < allModels.length; i++){
                for (int j=0;j<allModels.length;j++){
                    if (i != j){
                        boolean isDifferent = significance.isDifferent(predictionMap.get(allModels[i]), predictionMap.get(allModels[j]));
                        logger.info(String.format("Model %s and Model %s are statistically %s", allModels[i], allModels[j], isDifferent ? "different" : "same"));
                    }

                }
            }
        }


    }
}
