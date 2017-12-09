package projetosweka;

import java.io.FileReader;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

// Algoritmo Árvore de decisão
// ID3 Funciona apenas para dados nominais
// J48 FUnciona para ambos

public class ÁrvoreDeDecisão 
{
    public static void main(String[] args) 
    {
        try
        {
            String file = "/home/marcelo/NetBeansProjects/ProjetosWeka/data/contact-lenses.arff";
            
            //FileReader fr = new FileReader(file); // Também serveria aqui
            DataSource ds = new DataSource( file );
            
            // Instances ins = new Instances(fr); // Seria assim se utiliza-se FileReader
            Instances ins = ds.getDataSet();
            
            int iNumAttributes = ins.numAttributes(); // Número de atributos
            int iNumInstances = ins.numInstances(); // Número de instâncias
            int iClass = iNumAttributes - 1;
            
            ins.setClassIndex( iClass ); // Mostra para o classificador qual é a classe
            
            Id3 id3 = new Id3( ); // Instancia árvore ID3
            J48 j48 = new J48( ); // Instancia árvore J48
            
            id3.buildClassifier( ins ); // Cria a árvore de decisão ID3
            j48.buildClassifier( ins ); // Cria a árvore de decisão J48
            
            System.out.println(id3); // Exibe a árvore criada pelo classificador ID3
            System.out.println(j48); // Exibe a árvore criada pelo classificador J48
        }
        catch(Exception e){ System.out.println(e.getMessage()); }
    }
}
