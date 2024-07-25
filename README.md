> Para mais detalhes acesse o PDF abaixo: <br>
[PDF](https://github.com/lorenzoppx/Algoritmos_Deformacao_Placas/blob/master/Apresentacao_Deformacao_de_Placas.pdf)
<br>
Partindo de um conhecimento prévio inicial, é conhecida a equação diferencial de quarta ordem que relaciona a deformação sofrida pela placa w, a carga a qual a placa é submetida P e a rigidez da placa D. <br>
<p align="center">
<img src="https://github.com/lorenzoppx/Algoritmos_Deformacao_Placas/blob/master/Equation.png" width="600">
<p />
Essa equação é conhecida como Equação Diferencial de Lagrange em coordenadas Retangulares. Como essa equação é de quarta ordem então são necessário a definição de duas condições de contorno para todo o bordo da placa. Serão apresentados apenas dois tipos de contorno, porém é importante salientar que não são os únicos. <br>
O stencil foi obtido através da resolução de um sistema utilizando da expensão de Taylor.Tomando que a função g é contínua pela expansão de Taylor, temos que, <br>
<p align="center">
<img src="https://github.com/lorenzoppx/Algoritmos_Deformacao_Placas/blob/master/Stencil.png" width="600">
<p />
Prévia dos resultados obtidos: <br>
<p align="center">
<img src="https://github.com/lorenzoppx/Algoritmos_Deformacao_Placas/blob/master/Resultado_1.png" width="600">
<p />
<p align="center">
<img src="https://github.com/lorenzoppx/Algoritmos_Deformacao_Placas/blob/master/Resultado_1_2.png" width="600">
<p />
