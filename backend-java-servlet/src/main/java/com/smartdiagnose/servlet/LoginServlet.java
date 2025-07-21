package com.smartdiagnose.servlet;

import javax.servlet.*;
import javax.servlet.http.*;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

public class LoginServlet extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp){
        Connection conexao = Conecta.conecta();
        String n = req.getParameter("usuario");
        String p = req.getParameter("senha");
        PreparedStatement ps = null;
        java.sql.ResultSet rs = null;
        try {
            PrintWriter out = resp.getWriter();
            ps = conexao.prepareStatement("Select nome from usuario where nome = ? and senha = ?");
            ps.setString(1,n);
            ps.setString(2,p);;
            rs = ps.executeQuery();
            if(rs.next()){
                RequestDispatcher rd = req.getRequestDispatcher("home.html");
                rd.forward(req,resp);
            }
            else{
                out.println("<font color=red size=18>login falhou!!");
                out.println("<a href=index.html>Try Again</a>");
            }
        } catch (SQLException | ServletException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}
