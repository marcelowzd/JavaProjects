package projetosweka;

import weka.classifiers.lazy.IBk;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

// Classificador VizinhoMaisProximo
// Dados contínuos

public class VizinhoMaisProximo 
{
    public static void main(String[] args) 
    {
        try
        {
            String file = "/home/marcelo/NetBeansProjects/ProjetosWeka/data/diabetes.arff";
            
            DataSource ds = new DataSource( file );
            
            Instances ins = ds.getDataSet(); // Método que armazena o dataset em uma variável
            
            int iNumAttributes = ins.numAttributes(); // Número de atributos
            int iClass = iNumAttributes - 1;
            
            ins.setClassIndex( iClass ); // Mostra para o classificador qual é a classe
            
            Instance in = new DenseInstance(iNumAttributes); // Instância que será classificada
            
            in.setDataset(ins); // Indica que a instãncia pertence ao dataset de instâncias
            
            double[] values = { 2, 197, 70, 45, 543, 30.5, 0.158, 53 }; // Valores da instância que
            // será classificada
            
            for(int i = 0; i < values.length; i++)
                in.setValue(i, values[i]); // Coloca os valores na instância
        
            IBk vmp = new IBk(); // Instância do vizinho mais próximo
            
            vmp.buildClassifier(ins); // Faz o classificador estudar as instâncias
            
            double classe = vmp.classifyInstance(in); // Classifica a instância que criamos
            in.setValue(iClass, classe); // Armazena a classe
            
            System.out.println("A classe foi: " + in.stringValue(iClass)); // Exibe a classe
        }
        catch(Exception e){ System.out.println(e.getMessage()); }
    }
}
