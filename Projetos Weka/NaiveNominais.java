package projetosweka;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

// Classificador NaiveBayes
// Dados nominais

public class NaiveNominais 
{
    public static void main(String[] args) 
    {
        try
        {        
            String file = "/home/marcelo/NetBeansProjects/ProjetosWeka/data/contact-lenses.arff";

            DataSource ds = new DataSource( file );

            Instances ins = ds.getDataSet(); // Método que armazena o dataset em uma variável

            int iNumAttributes = ins.numAttributes(); // Número de atributos
            int iClass = iNumAttributes - 1; // Atributo classe
            
            ins.setClassIndex(iClass);
            
            Instance in = new DenseInstance(iNumAttributes);
            in.setDataset(ins);
            
            String[] values = { "young", "hypermetrope", "yes", "normal" };
            
            for(int i = 0; i < values.length; i++)
                in.setValue(i, values[i]);
            
            NaiveBayes nb = new NaiveBayes();
            
            nb.buildClassifier(ins);
            
            double classe = nb.classifyInstance(in);
            in.setValue(iClass, classe);
            
            System.out.println("Classe determinada: " + in.stringValue(iClass));
        }
        catch(Exception e){ System.out.println(e.getMessage()); }
    }
}
