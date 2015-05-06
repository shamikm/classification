package org.maj.sm.ml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * @author shamik.majumdar
 */
public class DataService {
    private static final Logger LOGGER = LoggerFactory.getLogger(DataService.class);

    public Instances getData(String file){
        BufferedReader inputReader = null;

        try {
            InputStream is = this.getClass().getResourceAsStream(file);
            inputReader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            return new Instances(inputReader);
        } catch (Exception ex) {
            LOGGER.error(ex.getMessage(), ex);
        }
        return null;

    }

    public Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }
}
