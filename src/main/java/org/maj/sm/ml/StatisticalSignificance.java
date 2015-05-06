package org.maj.sm.ml;

import weka.core.FastVector;

/**
 * @author shamik.majumdar
 */
public interface StatisticalSignificance {
    boolean isDifferent(FastVector... predictions);
}
