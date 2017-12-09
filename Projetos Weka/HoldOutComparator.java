package projetosweka;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class HoldOutComparator 
{
    public static void main(String[] args) 
    {
        try
        {
            String file = "/home/marcelo/NetBeansProjects/ProjetosWeka/data/iris.arff";
            
            ConverterUtils.DataSource ds = new ConverterUtils.DataSource(file);
            
            Instances ins = ds.getDataSet(); // Método que armazena o dataset em uma variável
            
            int iNumAttributes = ins.numAttributes(); // Número de atributos
            int iNumInstances = ins.numInstances(); // Número de instâncias
            int iClass = iNumAttributes - 1; // Atributo classe
            
            ins.setClassIndex(iClass);
            
            Instances baseTreino = ins.trainCV(3, 0);
            Instances baseTeste = ins.testCV(3, 0);
            
            IBk kNN = new IBk(3);
            
            kNN.buildClassifier(ins);
            
            System.out.println("Real;KNN(3)");
            
            Instance teste;
            
            for(int i = 0; i < baseTeste.numInstances(); i++)
            {
                teste = baseTeste.instance(i);
                System.out.print(teste.stringValue(iClass) + ";");
                teste.setClassMissing();
                teste.setClassValue(kNN.classifyInstance(teste));
                System.out.println(teste.stringValue(iClass));
            }
        }
        catch(Exception e) { System.out.println(e.getMessage()); }
    }
}
