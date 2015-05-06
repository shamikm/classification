package org.maj.sm.ml;

import weka.classifiers.evaluation.NominalPrediction;
import weka.core.FastVector;

/**
 * Refer 6.3.1 of Algorithms of intelligent web
 * @author shamik.majumdar
 */
public class McNemarTest implements  StatisticalSignificance{
    // with 95% confidence
    private final double THRESHOLD  = 3.841;
    // with 90% confidence
    //private final double THRESHOLD  = 2.706D;

    @Override
    public boolean isDifferent(FastVector... predictions) {
        if (predictions != null && predictions.length == 2){
            return isDifferent(predictions[0],predictions[1]);
        }
        throw new IllegalArgumentException();
    }

    public boolean isDifferent(FastVector predOne, FastVector predOther){
        int size = predOne.size();
        int n11 = 0;
        int n10 = 0;
        int n01 = 0;
        int n00 = 0;

        for (int i=0; i < size; i++){
            NominalPrediction oneNP = (NominalPrediction) predOne.elementAt(i);
            NominalPrediction otherNP = (NominalPrediction) predOther.elementAt(i);
            if (rightPrediction(oneNP) && rightPrediction((otherNP))) {
                //if both classifiers are right
                n11++;
            }else if (rightPrediction(oneNP) && !rightPrediction((otherNP))){
                n10++;
            }else if (!rightPrediction(oneNP) && rightPrediction((otherNP))){
                n01++;
            }else {
                n00++;
            }
        }
        double a = Math.abs(n01-n10) - 1;
        double chi2 = a * a / (n01+n10);
        return chi2 > THRESHOLD;
    }

    private boolean rightPrediction(NominalPrediction np) {
        return np.actual() == np.predicted();
    }
}
