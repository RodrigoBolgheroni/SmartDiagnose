package com.smartdiagnose.servlet;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import io.github.cdimascio.dotenv.Dotenv;

public class Conecta {
    public static Connection conecta(){
        Connection conexao = null;
        try {
            Dotenv dotenv = Dotenv.load();

            String url = dotenv.get("DB_URL");
            String usuario = dotenv.get("DB_USER");
            String senha = dotenv.get("DB_PASSWORD");

            Class.forName("org.postgresql.Driver");
            conexao = DriverManager.getConnection(url, usuario, senha);
            ResultSet rsCliente = conexao.createStatement().executeQuery("SELECT * FROM usuario");
        } catch (ClassNotFoundException e) {
            System.out.println("Driver do banco de dados não localizado");
        } catch (SQLException ex) {
            System.out.println("Erro ao conectar no banco " + ex);
        } finally {
            if (conexao != null) {
                    System.out.println("Conexão Efetuada com sucesso!");
            }
        }
        return conexao;
    }

}
