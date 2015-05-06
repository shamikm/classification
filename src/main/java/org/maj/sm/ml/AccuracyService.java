package org.maj.sm.ml;

import weka.classifiers.evaluation.NominalPrediction;
import weka.core.FastVector;

/**
 * @author shamik.majumdar
 */
public class AccuracyService {
    public double calculateAccuracy(FastVector predictions) {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        return 100 * correct / predictions.size();
    }
}
