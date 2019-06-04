
import torch
import torch.nn as nn
import geoopt.manifolds.poincare.math as pmath
import geoopt
import hyrnn
import functools


## este codigo se basa en model.py que trae el hyrnn para implementar redes hyperbolicas
## el cual esta basado en como se deben implementar modelos de DL en pyTorch

class FFNN(nn.Module):
    ## creo un objeto que hereda las propiedades de 
    ## nn.Module que es un objeto que esta definido en
    ## torch, este objeto da un esquema para contruir nuevos 
    ## modelos
    ## todos ocupan esto, similar a Module in Keras
    ## donde igual se debe ocupar de la misma forma
    ## este debe definirse un init y un forward, pico idea q es lo ultimo

    def __init__(
        self,
        hidden_dim, # dim capa oculta
        project_dim, # dim en la projeccion en el disco de poincare
        use_distance_as_feature=True, ## no entiendo esto
        device=None, # dispositivo en que se correra, debe ser un cuda
        num_layers=1, ## numero de capas 
        num_classes=1, ## numero de clases mnist -> 10 clases (creo)
        c=1.0, ## curvatura !!:o 
        order=1, ## no se XD
    ):
        ## super sirve para correr el metodo __init__ de module, pero 
        ## tiene mas funciones cosmicas que no entiendo todavia
        super(FFNN, self).__init__()
        (cell_type, embedding_type, decision_type) = map(
            str.lower, [cell_type, embedding_type, decision_type]
        )
        

        ## le quite la parte del embedding porq no ocupamos texto

        ## desicion tpye antes, no se q signifique
        self.projector_source = hyrnn.MobiusLinear(
            hidden_dim, project_dim, c=c, order=order
        )
        self.projector_target = hyrnn.MobiusLinear(
            hidden_dim, project_dim, c=c, order=order
        )
        self.logits = hyrnn.MobiusDist2Hyperplane(project_dim, num_classes)

        ## esta parte no la entiendo
        self.ball = geoopt.PoincareBall(c).set_default_order(order)
        if use_distance_as_feature:
            if decision_type == "eucl":
                self.dist_bias = nn.Parameter(torch.zeros(project_dim))
            else:
                self.dist_bias = geoopt.ManifoldParameter(
                    torch.zeros(project_dim), manifold=self.ball
                )
        else:
            self.register_buffer("dist_bias", None)

        ## fin parte que no entiendo

        ## asignaciones cosmicas para que el modelo tenga 
        ## guardada las configuraciones
        self.decision_type = decision_type
        self.use_distance_as_feature = use_distance_as_feature
        self.device = device  # declaring device here due to fact we are using catalyst
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.c = c

        ## elimine cell type xq solo ocuparemos FFNN o moibiusLinear que creo q es la ffnn
        ## para no complicar mas el asunto
        self.cell = functools.partial(hyrnn.MobiusGRU, c=c, order=order)

        ## esta parte esta cosmica hace unas weas locas
        self.cell_source = self.cell(embedding_dim, self.hidden_dim, self.num_layers)
        self.cell_target = self.cell(embedding_dim, self.hidden_dim, self.num_layers)


    ## hasta aca llegue primer dia de trabajo
    def forward(self, input):
        source_input = input[0]
        target_input = input[1]
        alignment = input[2]
        batch_size = alignment.shape[0]

        source_input_data = self.embedding(source_input.data)
        target_input_data = self.embedding(target_input.data)

        zero_hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=self.device or source_input.device,
            dtype=source_input_data.dtype
        )

        ## aqui hace algo cosmimico hace un mapeo de euclidio al disco de poincare
        ## con exp, ta toda loca esta parte, hay q dejar el primer if creo (?)
        if self.embedding_type == "eucl" and "hyp" in self.cell_type:
            source_input_data = pmath.expmap0(source_input_data, c=self.c)
            target_input_data = pmath.expmap0(target_input_data, c=self.c)
        elif self.embedding_type == "hyp" and "eucl" in self.cell_type:
            source_input_data = pmath.logmap0(source_input_data, c=self.c)
            target_input_data = pmath.logmap0(target_input_data, c=self.c)
        # ht: (num_layers * num_directions, batch, hidden_size)

        source_input = torch.nn.utils.rnn.PackedSequence(
            source_input_data, source_input.batch_sizes
        )
        target_input = torch.nn.utils.rnn.PackedSequence(
            target_input_data, target_input.batch_sizes
        )

        _, source_hidden = self.cell_source(source_input, zero_hidden)
        _, target_hidden = self.cell_target(target_input, zero_hidden)

        # take hiddens from the last layer
        source_hidden = source_hidden[-1]
        target_hidden = target_hidden[-1][alignment]

        if self.decision_type == "hyp":
            if "eucl" in self.cell_type:
                source_hidden = pmath.expmap0(source_hidden, c=self.c)
                target_hidden = pmath.expmap0(target_hidden, c=self.c)
            source_projected = self.projector_source(source_hidden)
            target_projected = self.projector_target(target_hidden)
            projected = pmath.mobius_add(
                source_projected, target_projected, c=self.ball.c
            )
            if self.use_distance_as_feature:
                dist = (
                    pmath.dist(source_hidden, target_hidden, dim=-1, keepdim=True, c=self.ball.c) ** 2
                )
                bias = pmath.mobius_scalar_mul(dist, self.dist_bias, c=self.ball.c)
                projected = pmath.mobius_add(projected, bias, c=self.ball.c)
        else:
            if "hyp" in self.cell_type:
                source_hidden = pmath.logmap0(source_hidden, c=self.c)
                target_hidden = pmath.logmap0(target_hidden, c=self.c)
            projected = self.projector(
                torch.cat((source_hidden, target_hidden), dim=-1)
            )
            if self.use_distance_as_feature:
                dist = torch.sum(
                    (source_hidden - target_hidden).pow(2), dim=-1, keepdim=True
                )
                bias = self.dist_bias * dist
                projected = projected + bias

        logits = self.logits(projected)
        # CrossEntropy accepts logits
        return logits