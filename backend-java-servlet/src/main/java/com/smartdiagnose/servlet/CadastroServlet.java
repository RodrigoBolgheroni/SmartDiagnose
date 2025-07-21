package com.smartdiagnose.servlet;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class CadastroServlet extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp){
        Connection conexao = Conecta.conecta();
        String n = req.getParameter("usuario");
        String e = req.getParameter("email");
        String p = req.getParameter("senha");
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {
            ps = conexao.prepareStatement("INSERT INTO usuario(nome,email,senha) VALUES (?,?,?)");
            ps.setString(1,n);
            ps.setString(2,e);
            ps.setString(3,p);
            ps.executeUpdate();
            RequestDispatcher rd = req.getRequestDispatcher("index.html");
            rd.forward(req,resp);
        } catch (SQLException | ServletException | IOException ex) {
            throw new RuntimeException(ex);
        }

    }
}
