??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.12v2.2.0-57-g25fba035f38??
?
din/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namedin/batch_normalization/gamma
?
1din/batch_normalization/gamma/Read/ReadVariableOpReadVariableOpdin/batch_normalization/gamma*
_output_shapes
:*
dtype0
?
din/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namedin/batch_normalization/beta
?
0din/batch_normalization/beta/Read/ReadVariableOpReadVariableOpdin/batch_normalization/beta*
_output_shapes
:*
dtype0
?
#din/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#din/batch_normalization/moving_mean
?
7din/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp#din/batch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
'din/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'din/batch_normalization/moving_variance
?
;din/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp'din/batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
din/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_namedin/dense_6/kernel
y
&din/dense_6/kernel/Read/ReadVariableOpReadVariableOpdin/dense_6/kernel*
_output_shapes

:@*
dtype0
x
din/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedin/dense_6/bias
q
$din/dense_6/bias/Read/ReadVariableOpReadVariableOpdin/dense_6/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
din/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??'*)
shared_namedin/embedding/embeddings
?
,din/embedding/embeddings/Read/ReadVariableOpReadVariableOpdin/embedding/embeddings* 
_output_shapes
:
??'*
dtype0
?
!din/attention__layer/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: P*2
shared_name#!din/attention__layer/dense/kernel
?
5din/attention__layer/dense/kernel/Read/ReadVariableOpReadVariableOp!din/attention__layer/dense/kernel*
_output_shapes

: P*
dtype0
?
din/attention__layer/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*0
shared_name!din/attention__layer/dense/bias
?
3din/attention__layer/dense/bias/Read/ReadVariableOpReadVariableOpdin/attention__layer/dense/bias*
_output_shapes
:P*
dtype0
?
#din/attention__layer/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P(*4
shared_name%#din/attention__layer/dense_1/kernel
?
7din/attention__layer/dense_1/kernel/Read/ReadVariableOpReadVariableOp#din/attention__layer/dense_1/kernel*
_output_shapes

:P(*
dtype0
?
!din/attention__layer/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*2
shared_name#!din/attention__layer/dense_1/bias
?
5din/attention__layer/dense_1/bias/Read/ReadVariableOpReadVariableOp!din/attention__layer/dense_1/bias*
_output_shapes
:(*
dtype0
?
#din/attention__layer/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*4
shared_name%#din/attention__layer/dense_2/kernel
?
7din/attention__layer/dense_2/kernel/Read/ReadVariableOpReadVariableOp#din/attention__layer/dense_2/kernel*
_output_shapes

:(*
dtype0
?
!din/attention__layer/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!din/attention__layer/dense_2/bias
?
5din/attention__layer/dense_2/bias/Read/ReadVariableOpReadVariableOp!din/attention__layer/dense_2/bias*
_output_shapes
:*
dtype0
?
din/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_namedin/dense_3/kernel
z
&din/dense_3/kernel/Read/ReadVariableOpReadVariableOpdin/dense_3/kernel*
_output_shapes
:	?*
dtype0
y
din/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namedin/dense_3/bias
r
$din/dense_3/bias/Read/ReadVariableOpReadVariableOpdin/dense_3/bias*
_output_shapes	
:?*
dtype0
?
din/dense_3/p_re_lu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namedin/dense_3/p_re_lu/alpha
?
-din/dense_3/p_re_lu/alpha/Read/ReadVariableOpReadVariableOpdin/dense_3/p_re_lu/alpha*
_output_shapes	
:?*
dtype0
?
din/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_namedin/dense_4/kernel
{
&din/dense_4/kernel/Read/ReadVariableOpReadVariableOpdin/dense_4/kernel* 
_output_shapes
:
??*
dtype0
y
din/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namedin/dense_4/bias
r
$din/dense_4/bias/Read/ReadVariableOpReadVariableOpdin/dense_4/bias*
_output_shapes	
:?*
dtype0
?
din/dense_4/p_re_lu_1/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namedin/dense_4/p_re_lu_1/alpha
?
/din/dense_4/p_re_lu_1/alpha/Read/ReadVariableOpReadVariableOpdin/dense_4/p_re_lu_1/alpha*
_output_shapes	
:?*
dtype0
?
din/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*#
shared_namedin/dense_5/kernel
z
&din/dense_5/kernel/Read/ReadVariableOpReadVariableOpdin/dense_5/kernel*
_output_shapes
:	?@*
dtype0
x
din/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namedin/dense_5/bias
q
$din/dense_5/bias/Read/ReadVariableOpReadVariableOpdin/dense_5/bias*
_output_shapes
:@*
dtype0
?
din/dense_5/p_re_lu_2/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namedin/dense_5/p_re_lu_2/alpha
?
/din/dense_5/p_re_lu_2/alpha/Read/ReadVariableOpReadVariableOpdin/dense_5/p_re_lu_2/alpha*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
?
$Adam/din/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/din/batch_normalization/gamma/m
?
8Adam/din/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/din/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/din/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/din/batch_normalization/beta/m
?
7Adam/din/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp#Adam/din/batch_normalization/beta/m*
_output_shapes
:*
dtype0
?
Adam/din/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@**
shared_nameAdam/din/dense_6/kernel/m
?
-Adam/din/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_6/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/din/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/din/dense_6/bias/m

+Adam/din/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/din/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??'*0
shared_name!Adam/din/embedding/embeddings/m
?
3Adam/din/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/din/embedding/embeddings/m* 
_output_shapes
:
??'*
dtype0
?
(Adam/din/attention__layer/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: P*9
shared_name*(Adam/din/attention__layer/dense/kernel/m
?
<Adam/din/attention__layer/dense/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/din/attention__layer/dense/kernel/m*
_output_shapes

: P*
dtype0
?
&Adam/din/attention__layer/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*7
shared_name(&Adam/din/attention__layer/dense/bias/m
?
:Adam/din/attention__layer/dense/bias/m/Read/ReadVariableOpReadVariableOp&Adam/din/attention__layer/dense/bias/m*
_output_shapes
:P*
dtype0
?
*Adam/din/attention__layer/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P(*;
shared_name,*Adam/din/attention__layer/dense_1/kernel/m
?
>Adam/din/attention__layer/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/din/attention__layer/dense_1/kernel/m*
_output_shapes

:P(*
dtype0
?
(Adam/din/attention__layer/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*9
shared_name*(Adam/din/attention__layer/dense_1/bias/m
?
<Adam/din/attention__layer/dense_1/bias/m/Read/ReadVariableOpReadVariableOp(Adam/din/attention__layer/dense_1/bias/m*
_output_shapes
:(*
dtype0
?
*Adam/din/attention__layer/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*;
shared_name,*Adam/din/attention__layer/dense_2/kernel/m
?
>Adam/din/attention__layer/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/din/attention__layer/dense_2/kernel/m*
_output_shapes

:(*
dtype0
?
(Adam/din/attention__layer/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/din/attention__layer/dense_2/bias/m
?
<Adam/din/attention__layer/dense_2/bias/m/Read/ReadVariableOpReadVariableOp(Adam/din/attention__layer/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/din/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdam/din/dense_3/kernel/m
?
-Adam/din/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/din/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/din/dense_3/bias/m
?
+Adam/din/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_3/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/din/dense_3/p_re_lu/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/din/dense_3/p_re_lu/alpha/m
?
4Adam/din/dense_3/p_re_lu/alpha/m/Read/ReadVariableOpReadVariableOp Adam/din/dense_3/p_re_lu/alpha/m*
_output_shapes	
:?*
dtype0
?
Adam/din/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/din/dense_4/kernel/m
?
-Adam/din/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_4/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/din/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/din/dense_4/bias/m
?
+Adam/din/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_4/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/din/dense_4/p_re_lu_1/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/din/dense_4/p_re_lu_1/alpha/m
?
6Adam/din/dense_4/p_re_lu_1/alpha/m/Read/ReadVariableOpReadVariableOp"Adam/din/dense_4/p_re_lu_1/alpha/m*
_output_shapes	
:?*
dtype0
?
Adam/din/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@**
shared_nameAdam/din/dense_5/kernel/m
?
-Adam/din/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_5/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/din/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/din/dense_5/bias/m

+Adam/din/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/din/dense_5/bias/m*
_output_shapes
:@*
dtype0
?
"Adam/din/dense_5/p_re_lu_2/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/din/dense_5/p_re_lu_2/alpha/m
?
6Adam/din/dense_5/p_re_lu_2/alpha/m/Read/ReadVariableOpReadVariableOp"Adam/din/dense_5/p_re_lu_2/alpha/m*
_output_shapes
:@*
dtype0
?
$Adam/din/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/din/batch_normalization/gamma/v
?
8Adam/din/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/din/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/din/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/din/batch_normalization/beta/v
?
7Adam/din/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp#Adam/din/batch_normalization/beta/v*
_output_shapes
:*
dtype0
?
Adam/din/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@**
shared_nameAdam/din/dense_6/kernel/v
?
-Adam/din/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_6/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/din/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/din/dense_6/bias/v

+Adam/din/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/din/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??'*0
shared_name!Adam/din/embedding/embeddings/v
?
3Adam/din/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/din/embedding/embeddings/v* 
_output_shapes
:
??'*
dtype0
?
(Adam/din/attention__layer/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: P*9
shared_name*(Adam/din/attention__layer/dense/kernel/v
?
<Adam/din/attention__layer/dense/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/din/attention__layer/dense/kernel/v*
_output_shapes

: P*
dtype0
?
&Adam/din/attention__layer/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*7
shared_name(&Adam/din/attention__layer/dense/bias/v
?
:Adam/din/attention__layer/dense/bias/v/Read/ReadVariableOpReadVariableOp&Adam/din/attention__layer/dense/bias/v*
_output_shapes
:P*
dtype0
?
*Adam/din/attention__layer/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P(*;
shared_name,*Adam/din/attention__layer/dense_1/kernel/v
?
>Adam/din/attention__layer/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/din/attention__layer/dense_1/kernel/v*
_output_shapes

:P(*
dtype0
?
(Adam/din/attention__layer/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*9
shared_name*(Adam/din/attention__layer/dense_1/bias/v
?
<Adam/din/attention__layer/dense_1/bias/v/Read/ReadVariableOpReadVariableOp(Adam/din/attention__layer/dense_1/bias/v*
_output_shapes
:(*
dtype0
?
*Adam/din/attention__layer/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*;
shared_name,*Adam/din/attention__layer/dense_2/kernel/v
?
>Adam/din/attention__layer/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/din/attention__layer/dense_2/kernel/v*
_output_shapes

:(*
dtype0
?
(Adam/din/attention__layer/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/din/attention__layer/dense_2/bias/v
?
<Adam/din/attention__layer/dense_2/bias/v/Read/ReadVariableOpReadVariableOp(Adam/din/attention__layer/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/din/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdam/din/dense_3/kernel/v
?
-Adam/din/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/din/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/din/dense_3/bias/v
?
+Adam/din/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_3/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/din/dense_3/p_re_lu/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/din/dense_3/p_re_lu/alpha/v
?
4Adam/din/dense_3/p_re_lu/alpha/v/Read/ReadVariableOpReadVariableOp Adam/din/dense_3/p_re_lu/alpha/v*
_output_shapes	
:?*
dtype0
?
Adam/din/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/din/dense_4/kernel/v
?
-Adam/din/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_4/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/din/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/din/dense_4/bias/v
?
+Adam/din/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_4/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/din/dense_4/p_re_lu_1/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/din/dense_4/p_re_lu_1/alpha/v
?
6Adam/din/dense_4/p_re_lu_1/alpha/v/Read/ReadVariableOpReadVariableOp"Adam/din/dense_4/p_re_lu_1/alpha/v*
_output_shapes	
:?*
dtype0
?
Adam/din/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@**
shared_nameAdam/din/dense_5/kernel/v
?
-Adam/din/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_5/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/din/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/din/dense_5/bias/v

+Adam/din/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/din/dense_5/bias/v*
_output_shapes
:@*
dtype0
?
"Adam/din/dense_5/p_re_lu_2/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/din/dense_5/p_re_lu_2/alpha/v
?
6Adam/din/dense_5/p_re_lu_2/alpha/v/Read/ReadVariableOpReadVariableOp"Adam/din/dense_5/p_re_lu_2/alpha/v*
_output_shapes
:@*
dtype0

NoOpNoOp
?p
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?o
value?oB?o B?o
?
dense_feature_columns
sparse_feature_columns
embed_sparse_layers
embed_seq_layers
attention_layer
bn
ffn
dropout
	dense_final

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 

0
 

0
v
	att_dense
att_final_dense
regularization_losses
	variables
trainable_variables
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
 	keras_api

!0
"1
#2
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratem?m?(m?)m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?v?v?(v?)v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?
 
?
30
41
52
63
74
85
96
7
8
9
10
:11
;12
<13
=14
>15
?16
@17
A18
B19
(20
)21
?
30
41
52
63
74
85
96
7
8
:9
;10
<11
=12
>13
?14
@15
A16
B17
(18
)19
?
Clayer_regularization_losses
Dlayer_metrics
Enon_trainable_variables
Fmetrics
regularization_losses
	variables

Glayers
trainable_variables
 
 
b
3
embeddings
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api

L0
M1
h

8kernel
9bias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
 
*
40
51
62
73
84
95
*
40
51
62
73
84
95
?
Rlayer_regularization_losses
Slayer_metrics

Tlayers
Umetrics
regularization_losses
	variables
Vnon_trainable_variables
trainable_variables
 
VT
VARIABLE_VALUEdin/batch_normalization/gamma#bn/gamma/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdin/batch_normalization/beta"bn/beta/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE#din/batch_normalization/moving_mean)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE'din/batch_normalization/moving_variance-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

0
1
?
Wlayer_regularization_losses
Xlayer_metrics

Ylayers
Zmetrics
regularization_losses
	variables
[non_trainable_variables
trainable_variables
x
\
activation

:kernel
;bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
x
a
activation

=kernel
>bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
x
f
activation

@kernel
Abias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
 
 
 
?
klayer_regularization_losses
llayer_metrics

mlayers
nmetrics
$regularization_losses
%	variables
onon_trainable_variables
&trainable_variables
US
VARIABLE_VALUEdin/dense_6/kernel-dense_final/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEdin/dense_6/bias+dense_final/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
player_regularization_losses
qlayer_metrics

rlayers
smetrics
*regularization_losses
+	variables
tnon_trainable_variables
,trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdin/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!din/attention__layer/dense/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdin/attention__layer/dense/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#din/attention__layer/dense_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!din/attention__layer/dense_1/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#din/attention__layer/dense_2/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!din/attention__layer/dense_2/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdin/dense_3/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdin/dense_3/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdin/dense_3/p_re_lu/alpha'variables/13/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdin/dense_4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdin/dense_4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdin/dense_4/p_re_lu_1/alpha'variables/16/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdin/dense_5/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEdin/dense_5/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdin/dense_5/p_re_lu_2/alpha'variables/19/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

u0
v1
8
0
1
2
!3
"4
#5
6
	7
 

30

30
?
wlayer_regularization_losses
xlayer_metrics

ylayers
zmetrics
Hregularization_losses
I	variables
{non_trainable_variables
Jtrainable_variables
h

4kernel
5bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
l

6kernel
7bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 

80
91

80
91
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
Nregularization_losses
O	variables
?non_trainable_variables
Ptrainable_variables
 
 

L0
M1
2
 
 
 
 
 
 

0
1
a
	<alpha
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 

:0
;1
<2

:0
;1
<2
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
]regularization_losses
^	variables
?non_trainable_variables
_trainable_variables
a
	?alpha
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 

=0
>1
?2

=0
>1
?2
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
bregularization_losses
c	variables
?non_trainable_variables
dtrainable_variables
a
	Balpha
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 

@0
A1
B2

@0
A1
B2
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
gregularization_losses
h	variables
?non_trainable_variables
itrainable_variables
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
?
?
thresholds
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
 
 
 
 
 
 

40
51

40
51
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
|regularization_losses
}	variables
?non_trainable_variables
~trainable_variables
 

60
71

60
71
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
 
 
 
 
 
 

<0

<0
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
 
 

\0
 
 
 

?0

?0
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
 
 

a0
 
 
 

B0

B0
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
 
 

f0
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
yw
VARIABLE_VALUE$Adam/din/batch_normalization/gamma/m?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE#Adam/din/batch_normalization/beta/m>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/din/dense_6/kernel/mIdense_final/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/din/dense_6/bias/mGdense_final/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/din/embedding/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/din/attention__layer/dense/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/din/attention__layer/dense/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/din/attention__layer/dense_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/din/attention__layer/dense_1/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/din/attention__layer/dense_2/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/din/attention__layer/dense_2/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/din/dense_3/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/din/dense_3/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/din/dense_3/p_re_lu/alpha/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/din/dense_4/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/din/dense_4/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/din/dense_4/p_re_lu_1/alpha/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/din/dense_5/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/din/dense_5/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/din/dense_5/p_re_lu_2/alpha/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE$Adam/din/batch_normalization/gamma/v?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE#Adam/din/batch_normalization/beta/v>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/din/dense_6/kernel/vIdense_final/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/din/dense_6/bias/vGdense_final/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/din/embedding/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/din/attention__layer/dense/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/din/attention__layer/dense/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/din/attention__layer/dense_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/din/attention__layer/dense_1/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/din/attention__layer/dense_2/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE(Adam/din/attention__layer/dense_2/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/din/dense_3/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/din/dense_3/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/din/dense_3/p_re_lu/alpha/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/din/dense_4/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/din/dense_4/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/din/dense_4/p_re_lu_1/alpha/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/din/dense_5/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/din/dense_5/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/din/dense_5/p_re_lu_2/alpha/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_input_3Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4din/embedding/embeddings!din/attention__layer/dense/kerneldin/attention__layer/dense/bias#din/attention__layer/dense_1/kernel!din/attention__layer/dense_1/bias#din/attention__layer/dense_2/kernel!din/attention__layer/dense_2/bias#din/batch_normalization/moving_mean'din/batch_normalization/moving_variancedin/batch_normalization/betadin/batch_normalization/gammadin/dense_3/kerneldin/dense_3/biasdin/dense_3/p_re_lu/alphadin/dense_4/kerneldin/dense_4/biasdin/dense_4/p_re_lu_1/alphadin/dense_5/kerneldin/dense_5/biasdin/dense_5/p_re_lu_2/alphadin/dense_6/kerneldin/dense_6/bias*%
Tin
2*
Tout
2*'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_10168
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1din/batch_normalization/gamma/Read/ReadVariableOp0din/batch_normalization/beta/Read/ReadVariableOp7din/batch_normalization/moving_mean/Read/ReadVariableOp;din/batch_normalization/moving_variance/Read/ReadVariableOp&din/dense_6/kernel/Read/ReadVariableOp$din/dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp,din/embedding/embeddings/Read/ReadVariableOp5din/attention__layer/dense/kernel/Read/ReadVariableOp3din/attention__layer/dense/bias/Read/ReadVariableOp7din/attention__layer/dense_1/kernel/Read/ReadVariableOp5din/attention__layer/dense_1/bias/Read/ReadVariableOp7din/attention__layer/dense_2/kernel/Read/ReadVariableOp5din/attention__layer/dense_2/bias/Read/ReadVariableOp&din/dense_3/kernel/Read/ReadVariableOp$din/dense_3/bias/Read/ReadVariableOp-din/dense_3/p_re_lu/alpha/Read/ReadVariableOp&din/dense_4/kernel/Read/ReadVariableOp$din/dense_4/bias/Read/ReadVariableOp/din/dense_4/p_re_lu_1/alpha/Read/ReadVariableOp&din/dense_5/kernel/Read/ReadVariableOp$din/dense_5/bias/Read/ReadVariableOp/din/dense_5/p_re_lu_2/alpha/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp8Adam/din/batch_normalization/gamma/m/Read/ReadVariableOp7Adam/din/batch_normalization/beta/m/Read/ReadVariableOp-Adam/din/dense_6/kernel/m/Read/ReadVariableOp+Adam/din/dense_6/bias/m/Read/ReadVariableOp3Adam/din/embedding/embeddings/m/Read/ReadVariableOp<Adam/din/attention__layer/dense/kernel/m/Read/ReadVariableOp:Adam/din/attention__layer/dense/bias/m/Read/ReadVariableOp>Adam/din/attention__layer/dense_1/kernel/m/Read/ReadVariableOp<Adam/din/attention__layer/dense_1/bias/m/Read/ReadVariableOp>Adam/din/attention__layer/dense_2/kernel/m/Read/ReadVariableOp<Adam/din/attention__layer/dense_2/bias/m/Read/ReadVariableOp-Adam/din/dense_3/kernel/m/Read/ReadVariableOp+Adam/din/dense_3/bias/m/Read/ReadVariableOp4Adam/din/dense_3/p_re_lu/alpha/m/Read/ReadVariableOp-Adam/din/dense_4/kernel/m/Read/ReadVariableOp+Adam/din/dense_4/bias/m/Read/ReadVariableOp6Adam/din/dense_4/p_re_lu_1/alpha/m/Read/ReadVariableOp-Adam/din/dense_5/kernel/m/Read/ReadVariableOp+Adam/din/dense_5/bias/m/Read/ReadVariableOp6Adam/din/dense_5/p_re_lu_2/alpha/m/Read/ReadVariableOp8Adam/din/batch_normalization/gamma/v/Read/ReadVariableOp7Adam/din/batch_normalization/beta/v/Read/ReadVariableOp-Adam/din/dense_6/kernel/v/Read/ReadVariableOp+Adam/din/dense_6/bias/v/Read/ReadVariableOp3Adam/din/embedding/embeddings/v/Read/ReadVariableOp<Adam/din/attention__layer/dense/kernel/v/Read/ReadVariableOp:Adam/din/attention__layer/dense/bias/v/Read/ReadVariableOp>Adam/din/attention__layer/dense_1/kernel/v/Read/ReadVariableOp<Adam/din/attention__layer/dense_1/bias/v/Read/ReadVariableOp>Adam/din/attention__layer/dense_2/kernel/v/Read/ReadVariableOp<Adam/din/attention__layer/dense_2/bias/v/Read/ReadVariableOp-Adam/din/dense_3/kernel/v/Read/ReadVariableOp+Adam/din/dense_3/bias/v/Read/ReadVariableOp4Adam/din/dense_3/p_re_lu/alpha/v/Read/ReadVariableOp-Adam/din/dense_4/kernel/v/Read/ReadVariableOp+Adam/din/dense_4/bias/v/Read/ReadVariableOp6Adam/din/dense_4/p_re_lu_1/alpha/v/Read/ReadVariableOp-Adam/din/dense_5/kernel/v/Read/ReadVariableOp+Adam/din/dense_5/bias/v/Read/ReadVariableOp6Adam/din/dense_5/p_re_lu_2/alpha/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_11383
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedin/batch_normalization/gammadin/batch_normalization/beta#din/batch_normalization/moving_mean'din/batch_normalization/moving_variancedin/dense_6/kerneldin/dense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedin/embedding/embeddings!din/attention__layer/dense/kerneldin/attention__layer/dense/bias#din/attention__layer/dense_1/kernel!din/attention__layer/dense_1/bias#din/attention__layer/dense_2/kernel!din/attention__layer/dense_2/biasdin/dense_3/kerneldin/dense_3/biasdin/dense_3/p_re_lu/alphadin/dense_4/kerneldin/dense_4/biasdin/dense_4/p_re_lu_1/alphadin/dense_5/kerneldin/dense_5/biasdin/dense_5/p_re_lu_2/alphatotalcounttrue_positivestrue_negativesfalse_positivesfalse_negatives$Adam/din/batch_normalization/gamma/m#Adam/din/batch_normalization/beta/mAdam/din/dense_6/kernel/mAdam/din/dense_6/bias/mAdam/din/embedding/embeddings/m(Adam/din/attention__layer/dense/kernel/m&Adam/din/attention__layer/dense/bias/m*Adam/din/attention__layer/dense_1/kernel/m(Adam/din/attention__layer/dense_1/bias/m*Adam/din/attention__layer/dense_2/kernel/m(Adam/din/attention__layer/dense_2/bias/mAdam/din/dense_3/kernel/mAdam/din/dense_3/bias/m Adam/din/dense_3/p_re_lu/alpha/mAdam/din/dense_4/kernel/mAdam/din/dense_4/bias/m"Adam/din/dense_4/p_re_lu_1/alpha/mAdam/din/dense_5/kernel/mAdam/din/dense_5/bias/m"Adam/din/dense_5/p_re_lu_2/alpha/m$Adam/din/batch_normalization/gamma/v#Adam/din/batch_normalization/beta/vAdam/din/dense_6/kernel/vAdam/din/dense_6/bias/vAdam/din/embedding/embeddings/v(Adam/din/attention__layer/dense/kernel/v&Adam/din/attention__layer/dense/bias/v*Adam/din/attention__layer/dense_1/kernel/v(Adam/din/attention__layer/dense_1/bias/v*Adam/din/attention__layer/dense_2/kernel/v(Adam/din/attention__layer/dense_2/bias/vAdam/din/dense_3/kernel/vAdam/din/dense_3/bias/v Adam/din/dense_3/p_re_lu/alpha/vAdam/din/dense_4/kernel/vAdam/din/dense_4/bias/v"Adam/din/dense_4/p_re_lu_1/alpha/vAdam/din/dense_5/kernel/vAdam/din/dense_5/bias/v"Adam/din/dense_5/p_re_lu_2/alpha/v*U
TinN
L2J*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_11614??
?
?
0__inference_attention__layer_layer_call_fn_10850
inputs_0
inputs_1
inputs_2
inputs_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention__layer_layer_call_and_return_conditional_losses_95592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesn
l:?????????:?????????:?????????:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
w
__inference_loss_fn_0_10991G
Cdin_embedding_embeddings_regularizer_square_readvariableop_resource
identity??
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpCdin_embedding_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/addo
IdentityIdentity,din/embedding/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
??
?

>__inference_din_layer_call_and_return_conditional_losses_10617
inputs_0
inputs_1
inputs_2
inputs_36
2embedding_embedding_lookup_readvariableop_resource<
8attention__layer_dense_tensordot_readvariableop_resource:
6attention__layer_dense_biasadd_readvariableop_resource>
:attention__layer_dense_1_tensordot_readvariableop_resource<
8attention__layer_dense_1_biasadd_readvariableop_resource>
:attention__layer_dense_2_tensordot_readvariableop_resource<
8attention__layer_dense_2_biasadd_readvariableop_resource4
0batch_normalization_cast_readvariableop_resource6
2batch_normalization_cast_1_readvariableop_resource6
2batch_normalization_cast_2_readvariableop_resource6
2batch_normalization_cast_3_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource+
'dense_3_p_re_lu_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_4_p_re_lu_1_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource-
)dense_5_p_re_lu_2_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinputs_2strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceZ

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2

NotEqual/y
NotEqualNotEqualstrided_slice:output:0NotEqual/y:output:0*
T0*'
_output_shapes
:?????????2

NotEqualc
CastCastNotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
Cast?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputs_2strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1?
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02+
)embedding/embedding_lookup/ReadVariableOp?
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axis?
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0strided_slice_1:output:0(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identityq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/concat_dim?
concat/concatIdentity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
concat/concat
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputs_3strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
+embedding_1/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02-
+embedding_1/embedding_lookup/ReadVariableOp?
!embedding_1/embedding_lookup/axisConst*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_1/embedding_lookup/axis?
embedding_1/embedding_lookupGatherV23embedding_1/embedding_lookup/ReadVariableOp:value:0strided_slice_2:output:0*embedding_1/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identityu
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/concat_dim?
concat_1/concatIdentity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
concat_1/concat?
attention__layer/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2!
attention__layer/Tile/multiples?
attention__layer/TileTileconcat_1/concat:output:0(attention__layer/Tile/multiples:output:0*
T0*(
_output_shapes
:??????????2
attention__layer/Tile?
attention__layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2 
attention__layer/Reshape/shape?
attention__layer/ReshapeReshapeattention__layer/Tile:output:0'attention__layer/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/Reshape?
attention__layer/subSub!attention__layer/Reshape:output:0concat/concat:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/sub?
attention__layer/mulMul!attention__layer/Reshape:output:0concat/concat:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/mul?
attention__layer/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
attention__layer/concat/axis?
attention__layer/concatConcatV2!attention__layer/Reshape:output:0concat/concat:output:0attention__layer/sub:z:0attention__layer/mul:z:0%attention__layer/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
attention__layer/concat?
/attention__layer/dense/Tensordot/ReadVariableOpReadVariableOp8attention__layer_dense_tensordot_readvariableop_resource*
_output_shapes

: P*
dtype021
/attention__layer/dense/Tensordot/ReadVariableOp?
%attention__layer/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%attention__layer/dense/Tensordot/axes?
%attention__layer/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%attention__layer/dense/Tensordot/free?
&attention__layer/dense/Tensordot/ShapeShape attention__layer/concat:output:0*
T0*
_output_shapes
:2(
&attention__layer/dense/Tensordot/Shape?
.attention__layer/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense/Tensordot/GatherV2/axis?
)attention__layer/dense/Tensordot/GatherV2GatherV2/attention__layer/dense/Tensordot/Shape:output:0.attention__layer/dense/Tensordot/free:output:07attention__layer/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)attention__layer/dense/Tensordot/GatherV2?
0attention__layer/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense/Tensordot/GatherV2_1/axis?
+attention__layer/dense/Tensordot/GatherV2_1GatherV2/attention__layer/dense/Tensordot/Shape:output:0.attention__layer/dense/Tensordot/axes:output:09attention__layer/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+attention__layer/dense/Tensordot/GatherV2_1?
&attention__layer/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&attention__layer/dense/Tensordot/Const?
%attention__layer/dense/Tensordot/ProdProd2attention__layer/dense/Tensordot/GatherV2:output:0/attention__layer/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%attention__layer/dense/Tensordot/Prod?
(attention__layer/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(attention__layer/dense/Tensordot/Const_1?
'attention__layer/dense/Tensordot/Prod_1Prod4attention__layer/dense/Tensordot/GatherV2_1:output:01attention__layer/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'attention__layer/dense/Tensordot/Prod_1?
,attention__layer/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,attention__layer/dense/Tensordot/concat/axis?
'attention__layer/dense/Tensordot/concatConcatV2.attention__layer/dense/Tensordot/free:output:0.attention__layer/dense/Tensordot/axes:output:05attention__layer/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'attention__layer/dense/Tensordot/concat?
&attention__layer/dense/Tensordot/stackPack.attention__layer/dense/Tensordot/Prod:output:00attention__layer/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&attention__layer/dense/Tensordot/stack?
*attention__layer/dense/Tensordot/transpose	Transpose attention__layer/concat:output:00attention__layer/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2,
*attention__layer/dense/Tensordot/transpose?
(attention__layer/dense/Tensordot/ReshapeReshape.attention__layer/dense/Tensordot/transpose:y:0/attention__layer/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(attention__layer/dense/Tensordot/Reshape?
'attention__layer/dense/Tensordot/MatMulMatMul1attention__layer/dense/Tensordot/Reshape:output:07attention__layer/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2)
'attention__layer/dense/Tensordot/MatMul?
(attention__layer/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2*
(attention__layer/dense/Tensordot/Const_2?
.attention__layer/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense/Tensordot/concat_1/axis?
)attention__layer/dense/Tensordot/concat_1ConcatV22attention__layer/dense/Tensordot/GatherV2:output:01attention__layer/dense/Tensordot/Const_2:output:07attention__layer/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)attention__layer/dense/Tensordot/concat_1?
 attention__layer/dense/TensordotReshape1attention__layer/dense/Tensordot/MatMul:product:02attention__layer/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2"
 attention__layer/dense/Tensordot?
-attention__layer/dense/BiasAdd/ReadVariableOpReadVariableOp6attention__layer_dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02/
-attention__layer/dense/BiasAdd/ReadVariableOp?
attention__layer/dense/BiasAddBiasAdd)attention__layer/dense/Tensordot:output:05attention__layer/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2 
attention__layer/dense/BiasAdd?
attention__layer/dense/SigmoidSigmoid'attention__layer/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2 
attention__layer/dense/Sigmoid?
1attention__layer/dense_1/Tensordot/ReadVariableOpReadVariableOp:attention__layer_dense_1_tensordot_readvariableop_resource*
_output_shapes

:P(*
dtype023
1attention__layer/dense_1/Tensordot/ReadVariableOp?
'attention__layer/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2)
'attention__layer/dense_1/Tensordot/axes?
'attention__layer/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2)
'attention__layer/dense_1/Tensordot/free?
(attention__layer/dense_1/Tensordot/ShapeShape"attention__layer/dense/Sigmoid:y:0*
T0*
_output_shapes
:2*
(attention__layer/dense_1/Tensordot/Shape?
0attention__layer/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_1/Tensordot/GatherV2/axis?
+attention__layer/dense_1/Tensordot/GatherV2GatherV21attention__layer/dense_1/Tensordot/Shape:output:00attention__layer/dense_1/Tensordot/free:output:09attention__layer/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+attention__layer/dense_1/Tensordot/GatherV2?
2attention__layer/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2attention__layer/dense_1/Tensordot/GatherV2_1/axis?
-attention__layer/dense_1/Tensordot/GatherV2_1GatherV21attention__layer/dense_1/Tensordot/Shape:output:00attention__layer/dense_1/Tensordot/axes:output:0;attention__layer/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2/
-attention__layer/dense_1/Tensordot/GatherV2_1?
(attention__layer/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(attention__layer/dense_1/Tensordot/Const?
'attention__layer/dense_1/Tensordot/ProdProd4attention__layer/dense_1/Tensordot/GatherV2:output:01attention__layer/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2)
'attention__layer/dense_1/Tensordot/Prod?
*attention__layer/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*attention__layer/dense_1/Tensordot/Const_1?
)attention__layer/dense_1/Tensordot/Prod_1Prod6attention__layer/dense_1/Tensordot/GatherV2_1:output:03attention__layer/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2+
)attention__layer/dense_1/Tensordot/Prod_1?
.attention__layer/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense_1/Tensordot/concat/axis?
)attention__layer/dense_1/Tensordot/concatConcatV20attention__layer/dense_1/Tensordot/free:output:00attention__layer/dense_1/Tensordot/axes:output:07attention__layer/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)attention__layer/dense_1/Tensordot/concat?
(attention__layer/dense_1/Tensordot/stackPack0attention__layer/dense_1/Tensordot/Prod:output:02attention__layer/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2*
(attention__layer/dense_1/Tensordot/stack?
,attention__layer/dense_1/Tensordot/transpose	Transpose"attention__layer/dense/Sigmoid:y:02attention__layer/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2.
,attention__layer/dense_1/Tensordot/transpose?
*attention__layer/dense_1/Tensordot/ReshapeReshape0attention__layer/dense_1/Tensordot/transpose:y:01attention__layer/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2,
*attention__layer/dense_1/Tensordot/Reshape?
)attention__layer/dense_1/Tensordot/MatMulMatMul3attention__layer/dense_1/Tensordot/Reshape:output:09attention__layer/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)attention__layer/dense_1/Tensordot/MatMul?
*attention__layer/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2,
*attention__layer/dense_1/Tensordot/Const_2?
0attention__layer/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_1/Tensordot/concat_1/axis?
+attention__layer/dense_1/Tensordot/concat_1ConcatV24attention__layer/dense_1/Tensordot/GatherV2:output:03attention__layer/dense_1/Tensordot/Const_2:output:09attention__layer/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2-
+attention__layer/dense_1/Tensordot/concat_1?
"attention__layer/dense_1/TensordotReshape3attention__layer/dense_1/Tensordot/MatMul:product:04attention__layer/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????(2$
"attention__layer/dense_1/Tensordot?
/attention__layer/dense_1/BiasAdd/ReadVariableOpReadVariableOp8attention__layer_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype021
/attention__layer/dense_1/BiasAdd/ReadVariableOp?
 attention__layer/dense_1/BiasAddBiasAdd+attention__layer/dense_1/Tensordot:output:07attention__layer/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????(2"
 attention__layer/dense_1/BiasAdd?
 attention__layer/dense_1/SigmoidSigmoid)attention__layer/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????(2"
 attention__layer/dense_1/Sigmoid?
1attention__layer/dense_2/Tensordot/ReadVariableOpReadVariableOp:attention__layer_dense_2_tensordot_readvariableop_resource*
_output_shapes

:(*
dtype023
1attention__layer/dense_2/Tensordot/ReadVariableOp?
'attention__layer/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2)
'attention__layer/dense_2/Tensordot/axes?
'attention__layer/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2)
'attention__layer/dense_2/Tensordot/free?
(attention__layer/dense_2/Tensordot/ShapeShape$attention__layer/dense_1/Sigmoid:y:0*
T0*
_output_shapes
:2*
(attention__layer/dense_2/Tensordot/Shape?
0attention__layer/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_2/Tensordot/GatherV2/axis?
+attention__layer/dense_2/Tensordot/GatherV2GatherV21attention__layer/dense_2/Tensordot/Shape:output:00attention__layer/dense_2/Tensordot/free:output:09attention__layer/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+attention__layer/dense_2/Tensordot/GatherV2?
2attention__layer/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2attention__layer/dense_2/Tensordot/GatherV2_1/axis?
-attention__layer/dense_2/Tensordot/GatherV2_1GatherV21attention__layer/dense_2/Tensordot/Shape:output:00attention__layer/dense_2/Tensordot/axes:output:0;attention__layer/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2/
-attention__layer/dense_2/Tensordot/GatherV2_1?
(attention__layer/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(attention__layer/dense_2/Tensordot/Const?
'attention__layer/dense_2/Tensordot/ProdProd4attention__layer/dense_2/Tensordot/GatherV2:output:01attention__layer/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2)
'attention__layer/dense_2/Tensordot/Prod?
*attention__layer/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*attention__layer/dense_2/Tensordot/Const_1?
)attention__layer/dense_2/Tensordot/Prod_1Prod6attention__layer/dense_2/Tensordot/GatherV2_1:output:03attention__layer/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2+
)attention__layer/dense_2/Tensordot/Prod_1?
.attention__layer/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense_2/Tensordot/concat/axis?
)attention__layer/dense_2/Tensordot/concatConcatV20attention__layer/dense_2/Tensordot/free:output:00attention__layer/dense_2/Tensordot/axes:output:07attention__layer/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)attention__layer/dense_2/Tensordot/concat?
(attention__layer/dense_2/Tensordot/stackPack0attention__layer/dense_2/Tensordot/Prod:output:02attention__layer/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2*
(attention__layer/dense_2/Tensordot/stack?
,attention__layer/dense_2/Tensordot/transpose	Transpose$attention__layer/dense_1/Sigmoid:y:02attention__layer/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????(2.
,attention__layer/dense_2/Tensordot/transpose?
*attention__layer/dense_2/Tensordot/ReshapeReshape0attention__layer/dense_2/Tensordot/transpose:y:01attention__layer/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2,
*attention__layer/dense_2/Tensordot/Reshape?
)attention__layer/dense_2/Tensordot/MatMulMatMul3attention__layer/dense_2/Tensordot/Reshape:output:09attention__layer/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)attention__layer/dense_2/Tensordot/MatMul?
*attention__layer/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*attention__layer/dense_2/Tensordot/Const_2?
0attention__layer/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_2/Tensordot/concat_1/axis?
+attention__layer/dense_2/Tensordot/concat_1ConcatV24attention__layer/dense_2/Tensordot/GatherV2:output:03attention__layer/dense_2/Tensordot/Const_2:output:09attention__layer/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2-
+attention__layer/dense_2/Tensordot/concat_1?
"attention__layer/dense_2/TensordotReshape3attention__layer/dense_2/Tensordot/MatMul:product:04attention__layer/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2$
"attention__layer/dense_2/Tensordot?
/attention__layer/dense_2/BiasAdd/ReadVariableOpReadVariableOp8attention__layer_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/attention__layer/dense_2/BiasAdd/ReadVariableOp?
 attention__layer/dense_2/BiasAddBiasAdd+attention__layer/dense_2/Tensordot:output:07attention__layer/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2"
 attention__layer/dense_2/BiasAdd?
attention__layer/SqueezeSqueeze)attention__layer/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2
attention__layer/Squeeze?
 attention__layer/ones_like/ShapeShape!attention__layer/Squeeze:output:0*
T0*
_output_shapes
:2"
 attention__layer/ones_like/Shape?
 attention__layer/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 attention__layer/ones_like/Const?
attention__layer/ones_likeFill)attention__layer/ones_like/Shape:output:0)attention__layer/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/ones_likey
attention__layer/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
attention__layer/mul_1/y?
attention__layer/mul_1Mul#attention__layer/ones_like:output:0!attention__layer/mul_1/y:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/mul_1y
attention__layer/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
attention__layer/Equal/y?
attention__layer/EqualEqualCast:y:0!attention__layer/Equal/y:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/Equal?
attention__layer/SelectV2SelectV2attention__layer/Equal:z:0attention__layer/mul_1:z:0!attention__layer/Squeeze:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/SelectV2?
attention__layer/SoftmaxSoftmax"attention__layer/SelectV2:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/Softmax?
attention__layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
attention__layer/ExpandDims/dim?
attention__layer/ExpandDims
ExpandDims"attention__layer/Softmax:softmax:0(attention__layer/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/ExpandDims?
attention__layer/MatMulBatchMatMulV2$attention__layer/ExpandDims:output:0concat/concat:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/MatMul?
attention__layer/Squeeze_1Squeeze attention__layer/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2
attention__layer/Squeeze_1i
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV2#attention__layer/Squeeze_1:output:0concat_1/concat:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????2

concat_2?
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype02)
'batch_normalization/Cast/ReadVariableOp?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp?
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_2/ReadVariableOp?
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_3/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulconcat_2:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2?
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/add_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAdd?
dense_3/p_re_lu/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/Relu?
dense_3/p_re_lu/ReadVariableOpReadVariableOp'dense_3_p_re_lu_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/p_re_lu/ReadVariableOp
dense_3/p_re_lu/NegNeg&dense_3/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
dense_3/p_re_lu/Neg?
dense_3/p_re_lu/Neg_1Negdense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/Neg_1?
dense_3/p_re_lu/Relu_1Reludense_3/p_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/Relu_1?
dense_3/p_re_lu/mulMuldense_3/p_re_lu/Neg:y:0$dense_3/p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/mul?
dense_3/p_re_lu/addAddV2"dense_3/p_re_lu/Relu:activations:0dense_3/p_re_lu/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/add?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/p_re_lu/add:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAdd?
dense_4/p_re_lu_1/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/Relu?
 dense_4/p_re_lu_1/ReadVariableOpReadVariableOp)dense_4_p_re_lu_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_4/p_re_lu_1/ReadVariableOp?
dense_4/p_re_lu_1/NegNeg(dense_4/p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
dense_4/p_re_lu_1/Neg?
dense_4/p_re_lu_1/Neg_1Negdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/Neg_1?
dense_4/p_re_lu_1/Relu_1Reludense_4/p_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/Relu_1?
dense_4/p_re_lu_1/mulMuldense_4/p_re_lu_1/Neg:y:0&dense_4/p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/mul?
dense_4/p_re_lu_1/addAddV2$dense_4/p_re_lu_1/Relu:activations:0dense_4/p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/add?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/p_re_lu_1/add:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/BiasAdd?
dense_5/p_re_lu_2/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/Relu?
 dense_5/p_re_lu_2/ReadVariableOpReadVariableOp)dense_5_p_re_lu_2_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_5/p_re_lu_2/ReadVariableOp?
dense_5/p_re_lu_2/NegNeg(dense_5/p_re_lu_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
dense_5/p_re_lu_2/Neg?
dense_5/p_re_lu_2/Neg_1Negdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/Neg_1?
dense_5/p_re_lu_2/Relu_1Reludense_5/p_re_lu_2/Neg_1:y:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/Relu_1?
dense_5/p_re_lu_2/mulMuldense_5/p_re_lu_2/Neg:y:0&dense_5/p_re_lu_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/mul?
dense_5/p_re_lu_2/addAddV2$dense_5/p_re_lu_2/Relu:activations:0dense_5/p_re_lu_2/mul:z:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/add}
dropout/IdentityIdentitydense_5/p_re_lu_2/add:z:0*
T0*'
_output_shapes
:?????????@2
dropout/Identity?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldropout/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddi
SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/add_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:::::::::::::::::::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_din_layer_call_fn_10098
input_1
input_2
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*%
Tin
2*
Tout
2*'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_din_layer_call_and_return_conditional_losses_99992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_din_layer_call_fn_10669
inputs_0
inputs_1
inputs_2
inputs_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*%
Tin
2*
Tout
2*'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_din_layer_call_and_return_conditional_losses_99992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_3_layer_call_fn_11076

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_96492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?+
!__inference__traced_restore_11614
file_prefix2
.assignvariableop_din_batch_normalization_gamma3
/assignvariableop_1_din_batch_normalization_beta:
6assignvariableop_2_din_batch_normalization_moving_mean>
:assignvariableop_3_din_batch_normalization_moving_variance)
%assignvariableop_4_din_dense_6_kernel'
#assignvariableop_5_din_dense_6_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate0
,assignvariableop_11_din_embedding_embeddings9
5assignvariableop_12_din_attention__layer_dense_kernel7
3assignvariableop_13_din_attention__layer_dense_bias;
7assignvariableop_14_din_attention__layer_dense_1_kernel9
5assignvariableop_15_din_attention__layer_dense_1_bias;
7assignvariableop_16_din_attention__layer_dense_2_kernel9
5assignvariableop_17_din_attention__layer_dense_2_bias*
&assignvariableop_18_din_dense_3_kernel(
$assignvariableop_19_din_dense_3_bias1
-assignvariableop_20_din_dense_3_p_re_lu_alpha*
&assignvariableop_21_din_dense_4_kernel(
$assignvariableop_22_din_dense_4_bias3
/assignvariableop_23_din_dense_4_p_re_lu_1_alpha*
&assignvariableop_24_din_dense_5_kernel(
$assignvariableop_25_din_dense_5_bias3
/assignvariableop_26_din_dense_5_p_re_lu_2_alpha
assignvariableop_27_total
assignvariableop_28_count&
"assignvariableop_29_true_positives&
"assignvariableop_30_true_negatives'
#assignvariableop_31_false_positives'
#assignvariableop_32_false_negatives<
8assignvariableop_33_adam_din_batch_normalization_gamma_m;
7assignvariableop_34_adam_din_batch_normalization_beta_m1
-assignvariableop_35_adam_din_dense_6_kernel_m/
+assignvariableop_36_adam_din_dense_6_bias_m7
3assignvariableop_37_adam_din_embedding_embeddings_m@
<assignvariableop_38_adam_din_attention__layer_dense_kernel_m>
:assignvariableop_39_adam_din_attention__layer_dense_bias_mB
>assignvariableop_40_adam_din_attention__layer_dense_1_kernel_m@
<assignvariableop_41_adam_din_attention__layer_dense_1_bias_mB
>assignvariableop_42_adam_din_attention__layer_dense_2_kernel_m@
<assignvariableop_43_adam_din_attention__layer_dense_2_bias_m1
-assignvariableop_44_adam_din_dense_3_kernel_m/
+assignvariableop_45_adam_din_dense_3_bias_m8
4assignvariableop_46_adam_din_dense_3_p_re_lu_alpha_m1
-assignvariableop_47_adam_din_dense_4_kernel_m/
+assignvariableop_48_adam_din_dense_4_bias_m:
6assignvariableop_49_adam_din_dense_4_p_re_lu_1_alpha_m1
-assignvariableop_50_adam_din_dense_5_kernel_m/
+assignvariableop_51_adam_din_dense_5_bias_m:
6assignvariableop_52_adam_din_dense_5_p_re_lu_2_alpha_m<
8assignvariableop_53_adam_din_batch_normalization_gamma_v;
7assignvariableop_54_adam_din_batch_normalization_beta_v1
-assignvariableop_55_adam_din_dense_6_kernel_v/
+assignvariableop_56_adam_din_dense_6_bias_v7
3assignvariableop_57_adam_din_embedding_embeddings_v@
<assignvariableop_58_adam_din_attention__layer_dense_kernel_v>
:assignvariableop_59_adam_din_attention__layer_dense_bias_vB
>assignvariableop_60_adam_din_attention__layer_dense_1_kernel_v@
<assignvariableop_61_adam_din_attention__layer_dense_1_bias_vB
>assignvariableop_62_adam_din_attention__layer_dense_2_kernel_v@
<assignvariableop_63_adam_din_attention__layer_dense_2_bias_v1
-assignvariableop_64_adam_din_dense_3_kernel_v/
+assignvariableop_65_adam_din_dense_3_bias_v8
4assignvariableop_66_adam_din_dense_3_p_re_lu_alpha_v1
-assignvariableop_67_adam_din_dense_4_kernel_v/
+assignvariableop_68_adam_din_dense_4_bias_v:
6assignvariableop_69_adam_din_dense_4_p_re_lu_1_alpha_v1
-assignvariableop_70_adam_din_dense_5_kernel_v/
+assignvariableop_71_adam_din_dense_5_bias_v:
6assignvariableop_72_adam_din_dense_5_p_re_lu_2_alpha_v
identity_74??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?!
value?!B?!IB#bn/gamma/.ATTRIBUTES/VARIABLE_VALUEB"bn/beta/.ATTRIBUTES/VARIABLE_VALUEB)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB-dense_final/kernel/.ATTRIBUTES/VARIABLE_VALUEB+dense_final/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdense_final/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGdense_final/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdense_final/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGdense_final/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*W
dtypesM
K2I	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp.assignvariableop_din_batch_normalization_gammaIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_din_batch_normalization_betaIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp6assignvariableop_2_din_batch_normalization_moving_meanIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp:assignvariableop_3_din_batch_normalization_moving_varianceIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_din_dense_6_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_din_dense_6_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_din_embedding_embeddingsIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp5assignvariableop_12_din_attention__layer_dense_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp3assignvariableop_13_din_attention__layer_dense_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_din_attention__layer_dense_1_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_din_attention__layer_dense_1_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_din_attention__layer_dense_2_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp5assignvariableop_17_din_attention__layer_dense_2_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_din_dense_3_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_din_dense_3_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_din_dense_3_p_re_lu_alphaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_din_dense_4_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_din_dense_4_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_din_dense_4_p_re_lu_1_alphaIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_din_dense_5_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_din_dense_5_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp/assignvariableop_26_din_dense_5_p_re_lu_2_alphaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_true_positivesIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp"assignvariableop_30_true_negativesIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_false_positivesIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_false_negativesIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp8assignvariableop_33_adam_din_batch_normalization_gamma_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_din_batch_normalization_beta_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_din_dense_6_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_din_dense_6_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp3assignvariableop_37_adam_din_embedding_embeddings_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp<assignvariableop_38_adam_din_attention__layer_dense_kernel_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp:assignvariableop_39_adam_din_attention__layer_dense_bias_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp>assignvariableop_40_adam_din_attention__layer_dense_1_kernel_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp<assignvariableop_41_adam_din_attention__layer_dense_1_bias_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_din_attention__layer_dense_2_kernel_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp<assignvariableop_43_adam_din_attention__layer_dense_2_bias_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp-assignvariableop_44_adam_din_dense_3_kernel_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_din_dense_3_bias_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_din_dense_3_p_re_lu_alpha_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_din_dense_4_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_din_dense_4_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp6assignvariableop_49_adam_din_dense_4_p_re_lu_1_alpha_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp-assignvariableop_50_adam_din_dense_5_kernel_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_din_dense_5_bias_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp6assignvariableop_52_adam_din_dense_5_p_re_lu_2_alpha_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp8assignvariableop_53_adam_din_batch_normalization_gamma_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp7assignvariableop_54_adam_din_batch_normalization_beta_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adam_din_dense_6_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp+assignvariableop_56_adam_din_dense_6_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp3assignvariableop_57_adam_din_embedding_embeddings_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp<assignvariableop_58_adam_din_attention__layer_dense_kernel_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp:assignvariableop_59_adam_din_attention__layer_dense_bias_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp>assignvariableop_60_adam_din_attention__layer_dense_1_kernel_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp<assignvariableop_61_adam_din_attention__layer_dense_1_bias_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_din_attention__layer_dense_2_kernel_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp<assignvariableop_63_adam_din_attention__layer_dense_2_bias_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp-assignvariableop_64_adam_din_dense_3_kernel_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_din_dense_3_bias_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp4assignvariableop_66_adam_din_dense_3_p_re_lu_alpha_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp-assignvariableop_67_adam_din_dense_4_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_din_dense_4_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_din_dense_4_p_re_lu_1_alpha_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp-assignvariableop_70_adam_din_dense_5_kernel_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_din_dense_5_bias_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_din_dense_5_p_re_lu_2_alpha_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_73?
Identity_74IdentityIdentity_73:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_74"#
identity_74Identity_74:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: 
?
n
(__inference_p_re_lu_2_layer_call_fn_9369

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_p_re_lu_2_layer_call_and_return_conditional_losses_93612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
?)
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9262

inputs
assignmovingavg_9237
assignmovingavg_1_9243 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/9237*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9237*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/9237*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/9237*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9237AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/9237*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/9243*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9243*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9243*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/9243*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9243AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/9243*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_embedding_layer_call_and_return_conditional_losses_9404

inputs,
(embedding_lookup_readvariableop_resource
identity??
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/addz
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
|
'__inference_dense_6_layer_call_fn_10978

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_97852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_11016

inputs,
(embedding_lookup_readvariableop_resource
identity??
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/addz
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
A__inference_dense_5_layer_call_and_return_conditional_losses_9725

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource%
!p_re_lu_2_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddl
p_re_lu_2/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/Relu?
p_re_lu_2/ReadVariableOpReadVariableOp!p_re_lu_2_readvariableop_resource*
_output_shapes
:@*
dtype02
p_re_lu_2/ReadVariableOpl
p_re_lu_2/NegNeg p_re_lu_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
p_re_lu_2/Negm
p_re_lu_2/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/Neg_1s
p_re_lu_2/Relu_1Relup_re_lu_2/Neg_1:y:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/Relu_1?
p_re_lu_2/mulMulp_re_lu_2/Neg:y:0p_re_lu_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/mul?
p_re_lu_2/addAddV2p_re_lu_2/Relu:activations:0p_re_lu_2/mul:z:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/adde
IdentityIdentityp_re_lu_2/add:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
`
'__inference_dropout_layer_call_fn_10954

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_97572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

a
B__inference_dropout_layer_call_and_return_conditional_losses_10944

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
M__inference_batch_normalization_layer_call_and_return_conditional_losses_9295

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity??
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?)
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10886

inputs
assignmovingavg_10861
assignmovingavg_1_10867 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?%AssignMovingAvg_1/AssignSubVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst*(
_class
loc:@AssignMovingAvg/10861*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10861*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*(
_class
loc:@AssignMovingAvg/10861*
_output_shapes
:2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*(
_class
loc:@AssignMovingAvg/10861*
_output_shapes
:2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10861AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/10861*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst**
_class 
loc:@AssignMovingAvg_1/10867*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10867*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/10867*
_output_shapes
:2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/10867*
_output_shapes
:2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10867AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/10867*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?\
?
=__inference_din_layer_call_and_return_conditional_losses_9902
input_1
input_2
input_3
input_4
embedding_9828
attention__layer_9841
attention__layer_9843
attention__layer_9845
attention__layer_9847
attention__layer_9849
attention__layer_9851
batch_normalization_9856
batch_normalization_9858
batch_normalization_9860
batch_normalization_9862
dense_3_9865
dense_3_9867
dense_3_9869
dense_4_9872
dense_4_9874
dense_4_9876
dense_5_9879
dense_5_9881
dense_5_9883
dense_6_9887
dense_6_9889
identity??(attention__layer/StatefulPartitionedCall?+batch_normalization/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinput_3strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceZ

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2

NotEqual/y
NotEqualNotEqualstrided_slice:output:0NotEqual/y:output:0*
T0*'
_output_shapes
:?????????2

NotEqualc
CastCastNotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
Cast?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinput_3strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_9828*
Tin
2*
Tout
2*+
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94042#
!embedding/StatefulPartitionedCallq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/concat_dim?
concat/concatIdentity*embedding/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
concat/concat
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinput_4strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_9828*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94382%
#embedding_1/StatefulPartitionedCallu
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/concat_dim?
concat_1/concatIdentity,embedding_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
concat_1/concat?
(attention__layer/StatefulPartitionedCallStatefulPartitionedCallconcat_1/concat:output:0concat/concat:output:0concat/concat:output:0Cast:y:0attention__layer_9841attention__layer_9843attention__layer_9845attention__layer_9847attention__layer_9849attention__layer_9851*
Tin
2
*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention__layer_layer_call_and_return_conditional_losses_95592*
(attention__layer/StatefulPartitionedCalli
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV21attention__layer/StatefulPartitionedCall:output:0concat_1/concat:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????2

concat_2?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0batch_normalization_9856batch_normalization_9858batch_normalization_9860batch_normalization_9862*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_92952-
+batch_normalization/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_3_9865dense_3_9867dense_3_9869*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_96492!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_9872dense_4_9874dense_4_9876*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_96872!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_9879dense_5_9881dense_5_9883*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_97252!
dense_5/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_97622
dropout/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6_9887dense_6_9889*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_97852!
dense_6/StatefulPartitionedCally
SigmoidSigmoid(dense_6/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_9828* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/add?
IdentityIdentitySigmoid:y:0)^attention__layer/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::2T
(attention__layer/StatefulPartitionedCall(attention__layer/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_embedding_layer_call_and_return_conditional_losses_9438

inputs,
(embedding_lookup_readvariableop_resource
identity??
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/addv
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????::K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
#__inference_din_layer_call_fn_10721
inputs_0
inputs_1
inputs_2
inputs_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*%
Tin
2*
Tout
2*'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_din_layer_call_and_return_conditional_losses_99992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
n
(__inference_p_re_lu_1_layer_call_fn_9348

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:??????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_p_re_lu_1_layer_call_and_return_conditional_losses_93402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
A__inference_dense_3_layer_call_and_return_conditional_losses_9649

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddi
p_re_lu/ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu/Relu?
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*
_output_shapes	
:?*
dtype02
p_re_lu/ReadVariableOpg
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
p_re_lu/Negj
p_re_lu/Neg_1NegBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu/Neg_1n
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
p_re_lu/Relu_1?
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
p_re_lu/mul?
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*(
_output_shapes
:??????????2
p_re_lu/addd
IdentityIdentityp_re_lu/add:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_5_layer_call_fn_11134

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_97252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?

__inference__wrapped_model_9166
input_1
input_2
input_3
input_4:
6din_embedding_embedding_lookup_readvariableop_resource@
<din_attention__layer_dense_tensordot_readvariableop_resource>
:din_attention__layer_dense_biasadd_readvariableop_resourceB
>din_attention__layer_dense_1_tensordot_readvariableop_resource@
<din_attention__layer_dense_1_biasadd_readvariableop_resourceB
>din_attention__layer_dense_2_tensordot_readvariableop_resource@
<din_attention__layer_dense_2_biasadd_readvariableop_resource8
4din_batch_normalization_cast_readvariableop_resource:
6din_batch_normalization_cast_1_readvariableop_resource:
6din_batch_normalization_cast_2_readvariableop_resource:
6din_batch_normalization_cast_3_readvariableop_resource.
*din_dense_3_matmul_readvariableop_resource/
+din_dense_3_biasadd_readvariableop_resource/
+din_dense_3_p_re_lu_readvariableop_resource.
*din_dense_4_matmul_readvariableop_resource/
+din_dense_4_biasadd_readvariableop_resource1
-din_dense_4_p_re_lu_1_readvariableop_resource.
*din_dense_5_matmul_readvariableop_resource/
+din_dense_5_biasadd_readvariableop_resource1
-din_dense_5_p_re_lu_2_readvariableop_resource.
*din_dense_6_matmul_readvariableop_resource/
+din_dense_6_biasadd_readvariableop_resource
identity??
din/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
din/strided_slice/stack?
din/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
din/strided_slice/stack_1?
din/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
din/strided_slice/stack_2?
din/strided_sliceStridedSliceinput_3 din/strided_slice/stack:output:0"din/strided_slice/stack_1:output:0"din/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
din/strided_sliceb
din/NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2
din/NotEqual/y?
din/NotEqualNotEqualdin/strided_slice:output:0din/NotEqual/y:output:0*
T0*'
_output_shapes
:?????????2
din/NotEqualo
din/CastCastdin/NotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2

din/Cast?
din/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
din/strided_slice_1/stack?
din/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
din/strided_slice_1/stack_1?
din/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
din/strided_slice_1/stack_2?
din/strided_slice_1StridedSliceinput_3"din/strided_slice_1/stack:output:0$din/strided_slice_1/stack_1:output:0$din/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
din/strided_slice_1?
-din/embedding/embedding_lookup/ReadVariableOpReadVariableOp6din_embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02/
-din/embedding/embedding_lookup/ReadVariableOp?
#din/embedding/embedding_lookup/axisConst*@
_class6
42loc:@din/embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2%
#din/embedding/embedding_lookup/axis?
din/embedding/embedding_lookupGatherV25din/embedding/embedding_lookup/ReadVariableOp:value:0din/strided_slice_1:output:0,din/embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*@
_class6
42loc:@din/embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2 
din/embedding/embedding_lookup?
'din/embedding/embedding_lookup/IdentityIdentity'din/embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????2)
'din/embedding/embedding_lookup/Identityy
din/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
din/concat/concat_dim?
din/concat/concatIdentity0din/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
din/concat/concat?
din/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
din/strided_slice_2/stack?
din/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
din/strided_slice_2/stack_1?
din/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
din/strided_slice_2/stack_2?
din/strided_slice_2StridedSliceinput_4"din/strided_slice_2/stack:output:0$din/strided_slice_2/stack_1:output:0$din/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
din/strided_slice_2?
/din/embedding_1/embedding_lookup/ReadVariableOpReadVariableOp6din_embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype021
/din/embedding_1/embedding_lookup/ReadVariableOp?
%din/embedding_1/embedding_lookup/axisConst*B
_class8
64loc:@din/embedding_1/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2'
%din/embedding_1/embedding_lookup/axis?
 din/embedding_1/embedding_lookupGatherV27din/embedding_1/embedding_lookup/ReadVariableOp:value:0din/strided_slice_2:output:0.din/embedding_1/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*B
_class8
64loc:@din/embedding_1/embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2"
 din/embedding_1/embedding_lookup?
)din/embedding_1/embedding_lookup/IdentityIdentity)din/embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:?????????2+
)din/embedding_1/embedding_lookup/Identity}
din/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
din/concat_1/concat_dim?
din/concat_1/concatIdentity2din/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
din/concat_1/concat?
#din/attention__layer/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2%
#din/attention__layer/Tile/multiples?
din/attention__layer/TileTiledin/concat_1/concat:output:0,din/attention__layer/Tile/multiples:output:0*
T0*(
_output_shapes
:??????????2
din/attention__layer/Tile?
"din/attention__layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2$
"din/attention__layer/Reshape/shape?
din/attention__layer/ReshapeReshape"din/attention__layer/Tile:output:0+din/attention__layer/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
din/attention__layer/Reshape?
din/attention__layer/subSub%din/attention__layer/Reshape:output:0din/concat/concat:output:0*
T0*+
_output_shapes
:?????????2
din/attention__layer/sub?
din/attention__layer/mulMul%din/attention__layer/Reshape:output:0din/concat/concat:output:0*
T0*+
_output_shapes
:?????????2
din/attention__layer/mul?
 din/attention__layer/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 din/attention__layer/concat/axis?
din/attention__layer/concatConcatV2%din/attention__layer/Reshape:output:0din/concat/concat:output:0din/attention__layer/sub:z:0din/attention__layer/mul:z:0)din/attention__layer/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
din/attention__layer/concat?
3din/attention__layer/dense/Tensordot/ReadVariableOpReadVariableOp<din_attention__layer_dense_tensordot_readvariableop_resource*
_output_shapes

: P*
dtype025
3din/attention__layer/dense/Tensordot/ReadVariableOp?
)din/attention__layer/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2+
)din/attention__layer/dense/Tensordot/axes?
)din/attention__layer/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2+
)din/attention__layer/dense/Tensordot/free?
*din/attention__layer/dense/Tensordot/ShapeShape$din/attention__layer/concat:output:0*
T0*
_output_shapes
:2,
*din/attention__layer/dense/Tensordot/Shape?
2din/attention__layer/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2din/attention__layer/dense/Tensordot/GatherV2/axis?
-din/attention__layer/dense/Tensordot/GatherV2GatherV23din/attention__layer/dense/Tensordot/Shape:output:02din/attention__layer/dense/Tensordot/free:output:0;din/attention__layer/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2/
-din/attention__layer/dense/Tensordot/GatherV2?
4din/attention__layer/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4din/attention__layer/dense/Tensordot/GatherV2_1/axis?
/din/attention__layer/dense/Tensordot/GatherV2_1GatherV23din/attention__layer/dense/Tensordot/Shape:output:02din/attention__layer/dense/Tensordot/axes:output:0=din/attention__layer/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:21
/din/attention__layer/dense/Tensordot/GatherV2_1?
*din/attention__layer/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*din/attention__layer/dense/Tensordot/Const?
)din/attention__layer/dense/Tensordot/ProdProd6din/attention__layer/dense/Tensordot/GatherV2:output:03din/attention__layer/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2+
)din/attention__layer/dense/Tensordot/Prod?
,din/attention__layer/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,din/attention__layer/dense/Tensordot/Const_1?
+din/attention__layer/dense/Tensordot/Prod_1Prod8din/attention__layer/dense/Tensordot/GatherV2_1:output:05din/attention__layer/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2-
+din/attention__layer/dense/Tensordot/Prod_1?
0din/attention__layer/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0din/attention__layer/dense/Tensordot/concat/axis?
+din/attention__layer/dense/Tensordot/concatConcatV22din/attention__layer/dense/Tensordot/free:output:02din/attention__layer/dense/Tensordot/axes:output:09din/attention__layer/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+din/attention__layer/dense/Tensordot/concat?
*din/attention__layer/dense/Tensordot/stackPack2din/attention__layer/dense/Tensordot/Prod:output:04din/attention__layer/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2,
*din/attention__layer/dense/Tensordot/stack?
.din/attention__layer/dense/Tensordot/transpose	Transpose$din/attention__layer/concat:output:04din/attention__layer/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 20
.din/attention__layer/dense/Tensordot/transpose?
,din/attention__layer/dense/Tensordot/ReshapeReshape2din/attention__layer/dense/Tensordot/transpose:y:03din/attention__layer/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2.
,din/attention__layer/dense/Tensordot/Reshape?
+din/attention__layer/dense/Tensordot/MatMulMatMul5din/attention__layer/dense/Tensordot/Reshape:output:0;din/attention__layer/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2-
+din/attention__layer/dense/Tensordot/MatMul?
,din/attention__layer/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2.
,din/attention__layer/dense/Tensordot/Const_2?
2din/attention__layer/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2din/attention__layer/dense/Tensordot/concat_1/axis?
-din/attention__layer/dense/Tensordot/concat_1ConcatV26din/attention__layer/dense/Tensordot/GatherV2:output:05din/attention__layer/dense/Tensordot/Const_2:output:0;din/attention__layer/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2/
-din/attention__layer/dense/Tensordot/concat_1?
$din/attention__layer/dense/TensordotReshape5din/attention__layer/dense/Tensordot/MatMul:product:06din/attention__layer/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2&
$din/attention__layer/dense/Tensordot?
1din/attention__layer/dense/BiasAdd/ReadVariableOpReadVariableOp:din_attention__layer_dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype023
1din/attention__layer/dense/BiasAdd/ReadVariableOp?
"din/attention__layer/dense/BiasAddBiasAdd-din/attention__layer/dense/Tensordot:output:09din/attention__layer/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2$
"din/attention__layer/dense/BiasAdd?
"din/attention__layer/dense/SigmoidSigmoid+din/attention__layer/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2$
"din/attention__layer/dense/Sigmoid?
5din/attention__layer/dense_1/Tensordot/ReadVariableOpReadVariableOp>din_attention__layer_dense_1_tensordot_readvariableop_resource*
_output_shapes

:P(*
dtype027
5din/attention__layer/dense_1/Tensordot/ReadVariableOp?
+din/attention__layer/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2-
+din/attention__layer/dense_1/Tensordot/axes?
+din/attention__layer/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2-
+din/attention__layer/dense_1/Tensordot/free?
,din/attention__layer/dense_1/Tensordot/ShapeShape&din/attention__layer/dense/Sigmoid:y:0*
T0*
_output_shapes
:2.
,din/attention__layer/dense_1/Tensordot/Shape?
4din/attention__layer/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4din/attention__layer/dense_1/Tensordot/GatherV2/axis?
/din/attention__layer/dense_1/Tensordot/GatherV2GatherV25din/attention__layer/dense_1/Tensordot/Shape:output:04din/attention__layer/dense_1/Tensordot/free:output:0=din/attention__layer/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:21
/din/attention__layer/dense_1/Tensordot/GatherV2?
6din/attention__layer/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 28
6din/attention__layer/dense_1/Tensordot/GatherV2_1/axis?
1din/attention__layer/dense_1/Tensordot/GatherV2_1GatherV25din/attention__layer/dense_1/Tensordot/Shape:output:04din/attention__layer/dense_1/Tensordot/axes:output:0?din/attention__layer/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:23
1din/attention__layer/dense_1/Tensordot/GatherV2_1?
,din/attention__layer/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,din/attention__layer/dense_1/Tensordot/Const?
+din/attention__layer/dense_1/Tensordot/ProdProd8din/attention__layer/dense_1/Tensordot/GatherV2:output:05din/attention__layer/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2-
+din/attention__layer/dense_1/Tensordot/Prod?
.din/attention__layer/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.din/attention__layer/dense_1/Tensordot/Const_1?
-din/attention__layer/dense_1/Tensordot/Prod_1Prod:din/attention__layer/dense_1/Tensordot/GatherV2_1:output:07din/attention__layer/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2/
-din/attention__layer/dense_1/Tensordot/Prod_1?
2din/attention__layer/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2din/attention__layer/dense_1/Tensordot/concat/axis?
-din/attention__layer/dense_1/Tensordot/concatConcatV24din/attention__layer/dense_1/Tensordot/free:output:04din/attention__layer/dense_1/Tensordot/axes:output:0;din/attention__layer/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2/
-din/attention__layer/dense_1/Tensordot/concat?
,din/attention__layer/dense_1/Tensordot/stackPack4din/attention__layer/dense_1/Tensordot/Prod:output:06din/attention__layer/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2.
,din/attention__layer/dense_1/Tensordot/stack?
0din/attention__layer/dense_1/Tensordot/transpose	Transpose&din/attention__layer/dense/Sigmoid:y:06din/attention__layer/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P22
0din/attention__layer/dense_1/Tensordot/transpose?
.din/attention__layer/dense_1/Tensordot/ReshapeReshape4din/attention__layer/dense_1/Tensordot/transpose:y:05din/attention__layer/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????20
.din/attention__layer/dense_1/Tensordot/Reshape?
-din/attention__layer/dense_1/Tensordot/MatMulMatMul7din/attention__layer/dense_1/Tensordot/Reshape:output:0=din/attention__layer/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2/
-din/attention__layer/dense_1/Tensordot/MatMul?
.din/attention__layer/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(20
.din/attention__layer/dense_1/Tensordot/Const_2?
4din/attention__layer/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4din/attention__layer/dense_1/Tensordot/concat_1/axis?
/din/attention__layer/dense_1/Tensordot/concat_1ConcatV28din/attention__layer/dense_1/Tensordot/GatherV2:output:07din/attention__layer/dense_1/Tensordot/Const_2:output:0=din/attention__layer/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:21
/din/attention__layer/dense_1/Tensordot/concat_1?
&din/attention__layer/dense_1/TensordotReshape7din/attention__layer/dense_1/Tensordot/MatMul:product:08din/attention__layer/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????(2(
&din/attention__layer/dense_1/Tensordot?
3din/attention__layer/dense_1/BiasAdd/ReadVariableOpReadVariableOp<din_attention__layer_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype025
3din/attention__layer/dense_1/BiasAdd/ReadVariableOp?
$din/attention__layer/dense_1/BiasAddBiasAdd/din/attention__layer/dense_1/Tensordot:output:0;din/attention__layer/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????(2&
$din/attention__layer/dense_1/BiasAdd?
$din/attention__layer/dense_1/SigmoidSigmoid-din/attention__layer/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????(2&
$din/attention__layer/dense_1/Sigmoid?
5din/attention__layer/dense_2/Tensordot/ReadVariableOpReadVariableOp>din_attention__layer_dense_2_tensordot_readvariableop_resource*
_output_shapes

:(*
dtype027
5din/attention__layer/dense_2/Tensordot/ReadVariableOp?
+din/attention__layer/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2-
+din/attention__layer/dense_2/Tensordot/axes?
+din/attention__layer/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2-
+din/attention__layer/dense_2/Tensordot/free?
,din/attention__layer/dense_2/Tensordot/ShapeShape(din/attention__layer/dense_1/Sigmoid:y:0*
T0*
_output_shapes
:2.
,din/attention__layer/dense_2/Tensordot/Shape?
4din/attention__layer/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4din/attention__layer/dense_2/Tensordot/GatherV2/axis?
/din/attention__layer/dense_2/Tensordot/GatherV2GatherV25din/attention__layer/dense_2/Tensordot/Shape:output:04din/attention__layer/dense_2/Tensordot/free:output:0=din/attention__layer/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:21
/din/attention__layer/dense_2/Tensordot/GatherV2?
6din/attention__layer/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 28
6din/attention__layer/dense_2/Tensordot/GatherV2_1/axis?
1din/attention__layer/dense_2/Tensordot/GatherV2_1GatherV25din/attention__layer/dense_2/Tensordot/Shape:output:04din/attention__layer/dense_2/Tensordot/axes:output:0?din/attention__layer/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:23
1din/attention__layer/dense_2/Tensordot/GatherV2_1?
,din/attention__layer/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2.
,din/attention__layer/dense_2/Tensordot/Const?
+din/attention__layer/dense_2/Tensordot/ProdProd8din/attention__layer/dense_2/Tensordot/GatherV2:output:05din/attention__layer/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2-
+din/attention__layer/dense_2/Tensordot/Prod?
.din/attention__layer/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.din/attention__layer/dense_2/Tensordot/Const_1?
-din/attention__layer/dense_2/Tensordot/Prod_1Prod:din/attention__layer/dense_2/Tensordot/GatherV2_1:output:07din/attention__layer/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2/
-din/attention__layer/dense_2/Tensordot/Prod_1?
2din/attention__layer/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2din/attention__layer/dense_2/Tensordot/concat/axis?
-din/attention__layer/dense_2/Tensordot/concatConcatV24din/attention__layer/dense_2/Tensordot/free:output:04din/attention__layer/dense_2/Tensordot/axes:output:0;din/attention__layer/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2/
-din/attention__layer/dense_2/Tensordot/concat?
,din/attention__layer/dense_2/Tensordot/stackPack4din/attention__layer/dense_2/Tensordot/Prod:output:06din/attention__layer/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2.
,din/attention__layer/dense_2/Tensordot/stack?
0din/attention__layer/dense_2/Tensordot/transpose	Transpose(din/attention__layer/dense_1/Sigmoid:y:06din/attention__layer/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????(22
0din/attention__layer/dense_2/Tensordot/transpose?
.din/attention__layer/dense_2/Tensordot/ReshapeReshape4din/attention__layer/dense_2/Tensordot/transpose:y:05din/attention__layer/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????20
.din/attention__layer/dense_2/Tensordot/Reshape?
-din/attention__layer/dense_2/Tensordot/MatMulMatMul7din/attention__layer/dense_2/Tensordot/Reshape:output:0=din/attention__layer/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2/
-din/attention__layer/dense_2/Tensordot/MatMul?
.din/attention__layer/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:20
.din/attention__layer/dense_2/Tensordot/Const_2?
4din/attention__layer/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4din/attention__layer/dense_2/Tensordot/concat_1/axis?
/din/attention__layer/dense_2/Tensordot/concat_1ConcatV28din/attention__layer/dense_2/Tensordot/GatherV2:output:07din/attention__layer/dense_2/Tensordot/Const_2:output:0=din/attention__layer/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:21
/din/attention__layer/dense_2/Tensordot/concat_1?
&din/attention__layer/dense_2/TensordotReshape7din/attention__layer/dense_2/Tensordot/MatMul:product:08din/attention__layer/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2(
&din/attention__layer/dense_2/Tensordot?
3din/attention__layer/dense_2/BiasAdd/ReadVariableOpReadVariableOp<din_attention__layer_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3din/attention__layer/dense_2/BiasAdd/ReadVariableOp?
$din/attention__layer/dense_2/BiasAddBiasAdd/din/attention__layer/dense_2/Tensordot:output:0;din/attention__layer/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2&
$din/attention__layer/dense_2/BiasAdd?
din/attention__layer/SqueezeSqueeze-din/attention__layer/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2
din/attention__layer/Squeeze?
$din/attention__layer/ones_like/ShapeShape%din/attention__layer/Squeeze:output:0*
T0*
_output_shapes
:2&
$din/attention__layer/ones_like/Shape?
$din/attention__layer/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$din/attention__layer/ones_like/Const?
din/attention__layer/ones_likeFill-din/attention__layer/ones_like/Shape:output:0-din/attention__layer/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2 
din/attention__layer/ones_like?
din/attention__layer/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
din/attention__layer/mul_1/y?
din/attention__layer/mul_1Mul'din/attention__layer/ones_like:output:0%din/attention__layer/mul_1/y:output:0*
T0*'
_output_shapes
:?????????2
din/attention__layer/mul_1?
din/attention__layer/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
din/attention__layer/Equal/y?
din/attention__layer/EqualEqualdin/Cast:y:0%din/attention__layer/Equal/y:output:0*
T0*'
_output_shapes
:?????????2
din/attention__layer/Equal?
din/attention__layer/SelectV2SelectV2din/attention__layer/Equal:z:0din/attention__layer/mul_1:z:0%din/attention__layer/Squeeze:output:0*
T0*'
_output_shapes
:?????????2
din/attention__layer/SelectV2?
din/attention__layer/SoftmaxSoftmax&din/attention__layer/SelectV2:output:0*
T0*'
_output_shapes
:?????????2
din/attention__layer/Softmax?
#din/attention__layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#din/attention__layer/ExpandDims/dim?
din/attention__layer/ExpandDims
ExpandDims&din/attention__layer/Softmax:softmax:0,din/attention__layer/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2!
din/attention__layer/ExpandDims?
din/attention__layer/MatMulBatchMatMulV2(din/attention__layer/ExpandDims:output:0din/concat/concat:output:0*
T0*+
_output_shapes
:?????????2
din/attention__layer/MatMul?
din/attention__layer/Squeeze_1Squeeze$din/attention__layer/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2 
din/attention__layer/Squeeze_1q
din/concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
din/concat_2/axis?
din/concat_2ConcatV2'din/attention__layer/Squeeze_1:output:0din/concat_1/concat:output:0din/concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
din/concat_2?
+din/batch_normalization/Cast/ReadVariableOpReadVariableOp4din_batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype02-
+din/batch_normalization/Cast/ReadVariableOp?
-din/batch_normalization/Cast_1/ReadVariableOpReadVariableOp6din_batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02/
-din/batch_normalization/Cast_1/ReadVariableOp?
-din/batch_normalization/Cast_2/ReadVariableOpReadVariableOp6din_batch_normalization_cast_2_readvariableop_resource*
_output_shapes
:*
dtype02/
-din/batch_normalization/Cast_2/ReadVariableOp?
-din/batch_normalization/Cast_3/ReadVariableOpReadVariableOp6din_batch_normalization_cast_3_readvariableop_resource*
_output_shapes
:*
dtype02/
-din/batch_normalization/Cast_3/ReadVariableOp?
'din/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2)
'din/batch_normalization/batchnorm/add/y?
%din/batch_normalization/batchnorm/addAddV25din/batch_normalization/Cast_1/ReadVariableOp:value:00din/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2'
%din/batch_normalization/batchnorm/add?
'din/batch_normalization/batchnorm/RsqrtRsqrt)din/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2)
'din/batch_normalization/batchnorm/Rsqrt?
%din/batch_normalization/batchnorm/mulMul+din/batch_normalization/batchnorm/Rsqrt:y:05din/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2'
%din/batch_normalization/batchnorm/mul?
'din/batch_normalization/batchnorm/mul_1Muldin/concat_2:output:0)din/batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2)
'din/batch_normalization/batchnorm/mul_1?
'din/batch_normalization/batchnorm/mul_2Mul3din/batch_normalization/Cast/ReadVariableOp:value:0)din/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2)
'din/batch_normalization/batchnorm/mul_2?
%din/batch_normalization/batchnorm/subSub5din/batch_normalization/Cast_2/ReadVariableOp:value:0+din/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2'
%din/batch_normalization/batchnorm/sub?
'din/batch_normalization/batchnorm/add_1AddV2+din/batch_normalization/batchnorm/mul_1:z:0)din/batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2)
'din/batch_normalization/batchnorm/add_1?
!din/dense_3/MatMul/ReadVariableOpReadVariableOp*din_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!din/dense_3/MatMul/ReadVariableOp?
din/dense_3/MatMulMatMul+din/batch_normalization/batchnorm/add_1:z:0)din/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
din/dense_3/MatMul?
"din/dense_3/BiasAdd/ReadVariableOpReadVariableOp+din_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"din/dense_3/BiasAdd/ReadVariableOp?
din/dense_3/BiasAddBiasAdddin/dense_3/MatMul:product:0*din/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
din/dense_3/BiasAdd?
din/dense_3/p_re_lu/ReluReludin/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
din/dense_3/p_re_lu/Relu?
"din/dense_3/p_re_lu/ReadVariableOpReadVariableOp+din_dense_3_p_re_lu_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"din/dense_3/p_re_lu/ReadVariableOp?
din/dense_3/p_re_lu/NegNeg*din/dense_3/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
din/dense_3/p_re_lu/Neg?
din/dense_3/p_re_lu/Neg_1Negdin/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
din/dense_3/p_re_lu/Neg_1?
din/dense_3/p_re_lu/Relu_1Reludin/dense_3/p_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
din/dense_3/p_re_lu/Relu_1?
din/dense_3/p_re_lu/mulMuldin/dense_3/p_re_lu/Neg:y:0(din/dense_3/p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
din/dense_3/p_re_lu/mul?
din/dense_3/p_re_lu/addAddV2&din/dense_3/p_re_lu/Relu:activations:0din/dense_3/p_re_lu/mul:z:0*
T0*(
_output_shapes
:??????????2
din/dense_3/p_re_lu/add?
!din/dense_4/MatMul/ReadVariableOpReadVariableOp*din_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!din/dense_4/MatMul/ReadVariableOp?
din/dense_4/MatMulMatMuldin/dense_3/p_re_lu/add:z:0)din/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
din/dense_4/MatMul?
"din/dense_4/BiasAdd/ReadVariableOpReadVariableOp+din_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"din/dense_4/BiasAdd/ReadVariableOp?
din/dense_4/BiasAddBiasAdddin/dense_4/MatMul:product:0*din/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
din/dense_4/BiasAdd?
din/dense_4/p_re_lu_1/ReluReludin/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
din/dense_4/p_re_lu_1/Relu?
$din/dense_4/p_re_lu_1/ReadVariableOpReadVariableOp-din_dense_4_p_re_lu_1_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$din/dense_4/p_re_lu_1/ReadVariableOp?
din/dense_4/p_re_lu_1/NegNeg,din/dense_4/p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
din/dense_4/p_re_lu_1/Neg?
din/dense_4/p_re_lu_1/Neg_1Negdin/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
din/dense_4/p_re_lu_1/Neg_1?
din/dense_4/p_re_lu_1/Relu_1Reludin/dense_4/p_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
din/dense_4/p_re_lu_1/Relu_1?
din/dense_4/p_re_lu_1/mulMuldin/dense_4/p_re_lu_1/Neg:y:0*din/dense_4/p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
din/dense_4/p_re_lu_1/mul?
din/dense_4/p_re_lu_1/addAddV2(din/dense_4/p_re_lu_1/Relu:activations:0din/dense_4/p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:??????????2
din/dense_4/p_re_lu_1/add?
!din/dense_5/MatMul/ReadVariableOpReadVariableOp*din_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02#
!din/dense_5/MatMul/ReadVariableOp?
din/dense_5/MatMulMatMuldin/dense_4/p_re_lu_1/add:z:0)din/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
din/dense_5/MatMul?
"din/dense_5/BiasAdd/ReadVariableOpReadVariableOp+din_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"din/dense_5/BiasAdd/ReadVariableOp?
din/dense_5/BiasAddBiasAdddin/dense_5/MatMul:product:0*din/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
din/dense_5/BiasAdd?
din/dense_5/p_re_lu_2/ReluReludin/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
din/dense_5/p_re_lu_2/Relu?
$din/dense_5/p_re_lu_2/ReadVariableOpReadVariableOp-din_dense_5_p_re_lu_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$din/dense_5/p_re_lu_2/ReadVariableOp?
din/dense_5/p_re_lu_2/NegNeg,din/dense_5/p_re_lu_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
din/dense_5/p_re_lu_2/Neg?
din/dense_5/p_re_lu_2/Neg_1Negdin/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
din/dense_5/p_re_lu_2/Neg_1?
din/dense_5/p_re_lu_2/Relu_1Reludin/dense_5/p_re_lu_2/Neg_1:y:0*
T0*'
_output_shapes
:?????????@2
din/dense_5/p_re_lu_2/Relu_1?
din/dense_5/p_re_lu_2/mulMuldin/dense_5/p_re_lu_2/Neg:y:0*din/dense_5/p_re_lu_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
din/dense_5/p_re_lu_2/mul?
din/dense_5/p_re_lu_2/addAddV2(din/dense_5/p_re_lu_2/Relu:activations:0din/dense_5/p_re_lu_2/mul:z:0*
T0*'
_output_shapes
:?????????@2
din/dense_5/p_re_lu_2/add?
din/dropout/IdentityIdentitydin/dense_5/p_re_lu_2/add:z:0*
T0*'
_output_shapes
:?????????@2
din/dropout/Identity?
!din/dense_6/MatMul/ReadVariableOpReadVariableOp*din_dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!din/dense_6/MatMul/ReadVariableOp?
din/dense_6/MatMulMatMuldin/dropout/Identity:output:0)din/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
din/dense_6/MatMul?
"din/dense_6/BiasAdd/ReadVariableOpReadVariableOp+din_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"din/dense_6/BiasAdd/ReadVariableOp?
din/dense_6/BiasAddBiasAdddin/dense_6/MatMul:product:0*din/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
din/dense_6/BiasAddu
din/SigmoidSigmoiddin/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
din/Sigmoidc
IdentityIdentitydin/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:::::::::::::::::::::::P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
}
A__inference_p_re_lu_layer_call_and_return_conditional_losses_9319

inputs
readvariableop_resource
identity?W
ReluReluinputs*
T0*0
_output_shapes
:??????????????????2
Reluu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOpO
NegNegReadVariableOp:value:0*
T0*
_output_shapes	
:?2
NegX
Neg_1Neginputs*
T0*0
_output_shapes
:??????????????????2
Neg_1^
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:??????????????????2
Relu_1c
mulMulNeg:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mulc
addAddV2Relu:activations:0mul:z:0*
T0*(
_output_shapes
:??????????2
add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????::X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10906

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity??
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

C__inference_p_re_lu_2_layer_call_and_return_conditional_losses_9361

inputs
readvariableop_resource
identity?W
ReluReluinputs*
T0*0
_output_shapes
:??????????????????2
Relut
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpN
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:@2
NegX
Neg_1Neginputs*
T0*0
_output_shapes
:??????????????????2
Neg_1^
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:??????????????????2
Relu_1b
mulMulNeg:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
mulb
addAddV2Relu:activations:0mul:z:0*
T0*'
_output_shapes
:?????????@2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????::X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
?|
?
J__inference_attention__layer_layer_call_and_return_conditional_losses_9559

inputs
inputs_1
inputs_2
inputs_3+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity?q
Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
Tile/multiplesh
TileTileinputsTile/multiples:output:0*
T0*(
_output_shapes
:??????????2
Tiles
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapez
ReshapeReshapeTile:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapec
subSubReshape:output:0inputs_1*
T0*+
_output_shapes
:?????????2
subc
mulMulReshape:output:0inputs_1*
T0*+
_output_shapes
:?????????2
mule
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2Reshape:output:0inputs_1sub:z:0mul:z:0concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concat?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: P*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freem
dense/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposeconcat:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense/BiasAddw
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
dense/Sigmoid?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:P(*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes?
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/frees
dense_1/Tensordot/ShapeShapedense/Sigmoid:y:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape?
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2?
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod?
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1?
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack?
dense_1/Tensordot/transpose	Transposedense/Sigmoid:y:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot/transpose?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_1/Tensordot/Const_2?
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????(2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????(2
dense_1/BiasAdd}
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????(2
dense_1/Sigmoid?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:(*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/freeu
dense_2/Tensordot/ShapeShapedense_1/Sigmoid:y:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposedense_1/Sigmoid:y:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????(2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_2/BiasAdd?
SqueezeSqueezedense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2	
Squeezeb
ones_like/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
	ones_likeW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_1/ym
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*'
_output_shapes
:?????????2
mul_1W
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
Equal/ye
EqualEqualinputs_3Equal/y:output:0*
T0*'
_output_shapes
:?????????2
Equalz
SelectV2SelectV2	Equal:z:0	mul_1:z:0Squeeze:output:0*
T0*'
_output_shapes
:?????????2

SelectV2b
SoftmaxSoftmaxSelectV2:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsv
MatMulBatchMatMulV2ExpandDims:output:0inputs_2*
T0*+
_output_shapes
:?????????2
MatMul{
	Squeeze_1SqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2
	Squeeze_1f
IdentityIdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesn
l:?????????:?????????:?????????:?????????:::::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
C
'__inference_dropout_layer_call_fn_10959

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_97622
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
o
)__inference_embedding_layer_call_fn_11023

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_10949

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
o
)__inference_embedding_layer_call_fn_11047

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?\
?
=__inference_din_layer_call_and_return_conditional_losses_9999

inputs
inputs_1
inputs_2
inputs_3
embedding_9925
attention__layer_9938
attention__layer_9940
attention__layer_9942
attention__layer_9944
attention__layer_9946
attention__layer_9948
batch_normalization_9953
batch_normalization_9955
batch_normalization_9957
batch_normalization_9959
dense_3_9962
dense_3_9964
dense_3_9966
dense_4_9969
dense_4_9971
dense_4_9973
dense_5_9976
dense_5_9978
dense_5_9980
dense_6_9984
dense_6_9986
identity??(attention__layer/StatefulPartitionedCall?+batch_normalization/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinputs_2strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceZ

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2

NotEqual/y
NotEqualNotEqualstrided_slice:output:0NotEqual/y:output:0*
T0*'
_output_shapes
:?????????2

NotEqualc
CastCastNotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
Cast?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputs_2strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_9925*
Tin
2*
Tout
2*+
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94042#
!embedding/StatefulPartitionedCallq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/concat_dim?
concat/concatIdentity*embedding/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
concat/concat
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputs_3strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_9925*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94382%
#embedding_1/StatefulPartitionedCallu
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/concat_dim?
concat_1/concatIdentity,embedding_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
concat_1/concat?
(attention__layer/StatefulPartitionedCallStatefulPartitionedCallconcat_1/concat:output:0concat/concat:output:0concat/concat:output:0Cast:y:0attention__layer_9938attention__layer_9940attention__layer_9942attention__layer_9944attention__layer_9946attention__layer_9948*
Tin
2
*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention__layer_layer_call_and_return_conditional_losses_95592*
(attention__layer/StatefulPartitionedCalli
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV21attention__layer/StatefulPartitionedCall:output:0concat_1/concat:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????2

concat_2?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0batch_normalization_9953batch_normalization_9955batch_normalization_9957batch_normalization_9959*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_92952-
+batch_normalization/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_3_9962dense_3_9964dense_3_9966*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_96492!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_9969dense_4_9971dense_4_9973*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_96872!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_9976dense_5_9978dense_5_9980*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_97252!
dense_5/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_97622
dropout/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6_9984dense_6_9986*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_97852!
dense_6/StatefulPartitionedCally
SigmoidSigmoid(dense_6/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_9925* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/add?
IdentityIdentitySigmoid:y:0)^attention__layer/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::2T
(attention__layer/StatefulPartitionedCall(attention__layer/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ǘ
?#
__inference__traced_save_11383
file_prefix<
8savev2_din_batch_normalization_gamma_read_readvariableop;
7savev2_din_batch_normalization_beta_read_readvariableopB
>savev2_din_batch_normalization_moving_mean_read_readvariableopF
Bsavev2_din_batch_normalization_moving_variance_read_readvariableop1
-savev2_din_dense_6_kernel_read_readvariableop/
+savev2_din_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop7
3savev2_din_embedding_embeddings_read_readvariableop@
<savev2_din_attention__layer_dense_kernel_read_readvariableop>
:savev2_din_attention__layer_dense_bias_read_readvariableopB
>savev2_din_attention__layer_dense_1_kernel_read_readvariableop@
<savev2_din_attention__layer_dense_1_bias_read_readvariableopB
>savev2_din_attention__layer_dense_2_kernel_read_readvariableop@
<savev2_din_attention__layer_dense_2_bias_read_readvariableop1
-savev2_din_dense_3_kernel_read_readvariableop/
+savev2_din_dense_3_bias_read_readvariableop8
4savev2_din_dense_3_p_re_lu_alpha_read_readvariableop1
-savev2_din_dense_4_kernel_read_readvariableop/
+savev2_din_dense_4_bias_read_readvariableop:
6savev2_din_dense_4_p_re_lu_1_alpha_read_readvariableop1
-savev2_din_dense_5_kernel_read_readvariableop/
+savev2_din_dense_5_bias_read_readvariableop:
6savev2_din_dense_5_p_re_lu_2_alpha_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableopC
?savev2_adam_din_batch_normalization_gamma_m_read_readvariableopB
>savev2_adam_din_batch_normalization_beta_m_read_readvariableop8
4savev2_adam_din_dense_6_kernel_m_read_readvariableop6
2savev2_adam_din_dense_6_bias_m_read_readvariableop>
:savev2_adam_din_embedding_embeddings_m_read_readvariableopG
Csavev2_adam_din_attention__layer_dense_kernel_m_read_readvariableopE
Asavev2_adam_din_attention__layer_dense_bias_m_read_readvariableopI
Esavev2_adam_din_attention__layer_dense_1_kernel_m_read_readvariableopG
Csavev2_adam_din_attention__layer_dense_1_bias_m_read_readvariableopI
Esavev2_adam_din_attention__layer_dense_2_kernel_m_read_readvariableopG
Csavev2_adam_din_attention__layer_dense_2_bias_m_read_readvariableop8
4savev2_adam_din_dense_3_kernel_m_read_readvariableop6
2savev2_adam_din_dense_3_bias_m_read_readvariableop?
;savev2_adam_din_dense_3_p_re_lu_alpha_m_read_readvariableop8
4savev2_adam_din_dense_4_kernel_m_read_readvariableop6
2savev2_adam_din_dense_4_bias_m_read_readvariableopA
=savev2_adam_din_dense_4_p_re_lu_1_alpha_m_read_readvariableop8
4savev2_adam_din_dense_5_kernel_m_read_readvariableop6
2savev2_adam_din_dense_5_bias_m_read_readvariableopA
=savev2_adam_din_dense_5_p_re_lu_2_alpha_m_read_readvariableopC
?savev2_adam_din_batch_normalization_gamma_v_read_readvariableopB
>savev2_adam_din_batch_normalization_beta_v_read_readvariableop8
4savev2_adam_din_dense_6_kernel_v_read_readvariableop6
2savev2_adam_din_dense_6_bias_v_read_readvariableop>
:savev2_adam_din_embedding_embeddings_v_read_readvariableopG
Csavev2_adam_din_attention__layer_dense_kernel_v_read_readvariableopE
Asavev2_adam_din_attention__layer_dense_bias_v_read_readvariableopI
Esavev2_adam_din_attention__layer_dense_1_kernel_v_read_readvariableopG
Csavev2_adam_din_attention__layer_dense_1_bias_v_read_readvariableopI
Esavev2_adam_din_attention__layer_dense_2_kernel_v_read_readvariableopG
Csavev2_adam_din_attention__layer_dense_2_bias_v_read_readvariableop8
4savev2_adam_din_dense_3_kernel_v_read_readvariableop6
2savev2_adam_din_dense_3_bias_v_read_readvariableop?
;savev2_adam_din_dense_3_p_re_lu_alpha_v_read_readvariableop8
4savev2_adam_din_dense_4_kernel_v_read_readvariableop6
2savev2_adam_din_dense_4_bias_v_read_readvariableopA
=savev2_adam_din_dense_4_p_re_lu_1_alpha_v_read_readvariableop8
4savev2_adam_din_dense_5_kernel_v_read_readvariableop6
2savev2_adam_din_dense_5_bias_v_read_readvariableopA
=savev2_adam_din_dense_5_p_re_lu_2_alpha_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e222af13f73b4b9bbc26c1c69a4e5a6a/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?!
value?!B?!IB#bn/gamma/.ATTRIBUTES/VARIABLE_VALUEB"bn/beta/.ATTRIBUTES/VARIABLE_VALUEB)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB-dense_final/kernel/.ATTRIBUTES/VARIABLE_VALUEB+dense_final/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIdense_final/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGdense_final/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBIdense_final/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGdense_final/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:I*
dtype0*?
value?B?IB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_din_batch_normalization_gamma_read_readvariableop7savev2_din_batch_normalization_beta_read_readvariableop>savev2_din_batch_normalization_moving_mean_read_readvariableopBsavev2_din_batch_normalization_moving_variance_read_readvariableop-savev2_din_dense_6_kernel_read_readvariableop+savev2_din_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_din_embedding_embeddings_read_readvariableop<savev2_din_attention__layer_dense_kernel_read_readvariableop:savev2_din_attention__layer_dense_bias_read_readvariableop>savev2_din_attention__layer_dense_1_kernel_read_readvariableop<savev2_din_attention__layer_dense_1_bias_read_readvariableop>savev2_din_attention__layer_dense_2_kernel_read_readvariableop<savev2_din_attention__layer_dense_2_bias_read_readvariableop-savev2_din_dense_3_kernel_read_readvariableop+savev2_din_dense_3_bias_read_readvariableop4savev2_din_dense_3_p_re_lu_alpha_read_readvariableop-savev2_din_dense_4_kernel_read_readvariableop+savev2_din_dense_4_bias_read_readvariableop6savev2_din_dense_4_p_re_lu_1_alpha_read_readvariableop-savev2_din_dense_5_kernel_read_readvariableop+savev2_din_dense_5_bias_read_readvariableop6savev2_din_dense_5_p_re_lu_2_alpha_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop?savev2_adam_din_batch_normalization_gamma_m_read_readvariableop>savev2_adam_din_batch_normalization_beta_m_read_readvariableop4savev2_adam_din_dense_6_kernel_m_read_readvariableop2savev2_adam_din_dense_6_bias_m_read_readvariableop:savev2_adam_din_embedding_embeddings_m_read_readvariableopCsavev2_adam_din_attention__layer_dense_kernel_m_read_readvariableopAsavev2_adam_din_attention__layer_dense_bias_m_read_readvariableopEsavev2_adam_din_attention__layer_dense_1_kernel_m_read_readvariableopCsavev2_adam_din_attention__layer_dense_1_bias_m_read_readvariableopEsavev2_adam_din_attention__layer_dense_2_kernel_m_read_readvariableopCsavev2_adam_din_attention__layer_dense_2_bias_m_read_readvariableop4savev2_adam_din_dense_3_kernel_m_read_readvariableop2savev2_adam_din_dense_3_bias_m_read_readvariableop;savev2_adam_din_dense_3_p_re_lu_alpha_m_read_readvariableop4savev2_adam_din_dense_4_kernel_m_read_readvariableop2savev2_adam_din_dense_4_bias_m_read_readvariableop=savev2_adam_din_dense_4_p_re_lu_1_alpha_m_read_readvariableop4savev2_adam_din_dense_5_kernel_m_read_readvariableop2savev2_adam_din_dense_5_bias_m_read_readvariableop=savev2_adam_din_dense_5_p_re_lu_2_alpha_m_read_readvariableop?savev2_adam_din_batch_normalization_gamma_v_read_readvariableop>savev2_adam_din_batch_normalization_beta_v_read_readvariableop4savev2_adam_din_dense_6_kernel_v_read_readvariableop2savev2_adam_din_dense_6_bias_v_read_readvariableop:savev2_adam_din_embedding_embeddings_v_read_readvariableopCsavev2_adam_din_attention__layer_dense_kernel_v_read_readvariableopAsavev2_adam_din_attention__layer_dense_bias_v_read_readvariableopEsavev2_adam_din_attention__layer_dense_1_kernel_v_read_readvariableopCsavev2_adam_din_attention__layer_dense_1_bias_v_read_readvariableopEsavev2_adam_din_attention__layer_dense_2_kernel_v_read_readvariableopCsavev2_adam_din_attention__layer_dense_2_bias_v_read_readvariableop4savev2_adam_din_dense_3_kernel_v_read_readvariableop2savev2_adam_din_dense_3_bias_v_read_readvariableop;savev2_adam_din_dense_3_p_re_lu_alpha_v_read_readvariableop4savev2_adam_din_dense_4_kernel_v_read_readvariableop2savev2_adam_din_dense_4_bias_v_read_readvariableop=savev2_adam_din_dense_4_p_re_lu_1_alpha_v_read_readvariableop4savev2_adam_din_dense_5_kernel_v_read_readvariableop2savev2_adam_din_dense_5_bias_v_read_readvariableop=savev2_adam_din_dense_5_p_re_lu_2_alpha_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *W
dtypesM
K2I	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::@:: : : : : :
??': P:P:P(:(:(::	?:?:?:
??:?:?:	?@:@:@: : :?:?:?:?:::@::
??': P:P:P(:(:(::	?:?:?:
??:?:?:	?@:@:@:::@::
??': P:P:P(:(:(::	?:?:?:
??:?:?:	?@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??':$ 

_output_shapes

: P: 

_output_shapes
:P:$ 

_output_shapes

:P(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:! 

_output_shapes	
:?:!!

_output_shapes	
:?: "

_output_shapes
:: #

_output_shapes
::$$ 

_output_shapes

:@: %

_output_shapes
::&&"
 
_output_shapes
:
??':$' 

_output_shapes

: P: (

_output_shapes
:P:$) 

_output_shapes

:P(: *

_output_shapes
:(:$+ 

_output_shapes

:(: ,

_output_shapes
::%-!

_output_shapes
:	?:!.

_output_shapes	
:?:!/

_output_shapes	
:?:&0"
 
_output_shapes
:
??:!1

_output_shapes	
:?:!2

_output_shapes	
:?:%3!

_output_shapes
:	?@: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
:: 7

_output_shapes
::$8 

_output_shapes

:@: 9

_output_shapes
::&:"
 
_output_shapes
:
??':$; 

_output_shapes

: P: <

_output_shapes
:P:$= 

_output_shapes

:P(: >

_output_shapes
:(:$? 

_output_shapes

:(: @

_output_shapes
::%A!

_output_shapes
:	?:!B

_output_shapes	
:?:!C

_output_shapes	
:?:&D"
 
_output_shapes
:
??:!E

_output_shapes	
:?:!F

_output_shapes	
:?:%G!

_output_shapes
:	?@: H

_output_shapes
:@: I

_output_shapes
:@:J

_output_shapes
: 
?

C__inference_p_re_lu_1_layer_call_and_return_conditional_losses_9340

inputs
readvariableop_resource
identity?W
ReluReluinputs*
T0*0
_output_shapes
:??????????????????2
Reluu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOpO
NegNegReadVariableOp:value:0*
T0*
_output_shapes	
:?2
NegX
Neg_1Neginputs*
T0*0
_output_shapes
:??????????????????2
Neg_1^
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:??????????????????2
Relu_1c
mulMulNeg:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mulc
addAddV2Relu:activations:0mul:z:0*
T0*(
_output_shapes
:??????????2
add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????::X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
B__inference_dense_6_layer_call_and_return_conditional_losses_10969

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_dense_4_layer_call_and_return_conditional_losses_9687

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
p_re_lu_1/ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/Relu?
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
p_re_lu_1/ReadVariableOpm
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
p_re_lu_1/Negn
p_re_lu_1/Neg_1NegBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/Neg_1t
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/Relu_1?
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/mul?
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/addf
IdentityIdentityp_re_lu_1/add:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_batch_normalization_layer_call_fn_10932

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_92952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

`
A__inference_dropout_layer_call_and_return_conditional_losses_9757

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?

>__inference_din_layer_call_and_return_conditional_losses_10404
inputs_0
inputs_1
inputs_2
inputs_36
2embedding_embedding_lookup_readvariableop_resource<
8attention__layer_dense_tensordot_readvariableop_resource:
6attention__layer_dense_biasadd_readvariableop_resource>
:attention__layer_dense_1_tensordot_readvariableop_resource<
8attention__layer_dense_1_biasadd_readvariableop_resource>
:attention__layer_dense_2_tensordot_readvariableop_resource<
8attention__layer_dense_2_biasadd_readvariableop_resource-
)batch_normalization_assignmovingavg_10314/
+batch_normalization_assignmovingavg_1_103204
0batch_normalization_cast_readvariableop_resource6
2batch_normalization_cast_1_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource+
'dense_3_p_re_lu_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource-
)dense_4_p_re_lu_1_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource-
)dense_5_p_re_lu_2_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??7batch_normalization/AssignMovingAvg/AssignSubVariableOp?9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinputs_2strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceZ

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2

NotEqual/y
NotEqualNotEqualstrided_slice:output:0NotEqual/y:output:0*
T0*'
_output_shapes
:?????????2

NotEqualc
CastCastNotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
Cast?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputs_2strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1?
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02+
)embedding/embedding_lookup/ReadVariableOp?
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axis?
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0strided_slice_1:output:0(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:?????????2
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identityq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/concat_dim?
concat/concatIdentity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
concat/concat
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputs_3strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
+embedding_1/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02-
+embedding_1/embedding_lookup/ReadVariableOp?
!embedding_1/embedding_lookup/axisConst*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_1/embedding_lookup/axis?
embedding_1/embedding_lookupGatherV23embedding_1/embedding_lookup/ReadVariableOp:value:0strided_slice_2:output:0*embedding_1/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identityu
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/concat_dim?
concat_1/concatIdentity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
concat_1/concat?
attention__layer/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2!
attention__layer/Tile/multiples?
attention__layer/TileTileconcat_1/concat:output:0(attention__layer/Tile/multiples:output:0*
T0*(
_output_shapes
:??????????2
attention__layer/Tile?
attention__layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2 
attention__layer/Reshape/shape?
attention__layer/ReshapeReshapeattention__layer/Tile:output:0'attention__layer/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/Reshape?
attention__layer/subSub!attention__layer/Reshape:output:0concat/concat:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/sub?
attention__layer/mulMul!attention__layer/Reshape:output:0concat/concat:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/mul?
attention__layer/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
attention__layer/concat/axis?
attention__layer/concatConcatV2!attention__layer/Reshape:output:0concat/concat:output:0attention__layer/sub:z:0attention__layer/mul:z:0%attention__layer/concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
attention__layer/concat?
/attention__layer/dense/Tensordot/ReadVariableOpReadVariableOp8attention__layer_dense_tensordot_readvariableop_resource*
_output_shapes

: P*
dtype021
/attention__layer/dense/Tensordot/ReadVariableOp?
%attention__layer/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%attention__layer/dense/Tensordot/axes?
%attention__layer/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%attention__layer/dense/Tensordot/free?
&attention__layer/dense/Tensordot/ShapeShape attention__layer/concat:output:0*
T0*
_output_shapes
:2(
&attention__layer/dense/Tensordot/Shape?
.attention__layer/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense/Tensordot/GatherV2/axis?
)attention__layer/dense/Tensordot/GatherV2GatherV2/attention__layer/dense/Tensordot/Shape:output:0.attention__layer/dense/Tensordot/free:output:07attention__layer/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)attention__layer/dense/Tensordot/GatherV2?
0attention__layer/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense/Tensordot/GatherV2_1/axis?
+attention__layer/dense/Tensordot/GatherV2_1GatherV2/attention__layer/dense/Tensordot/Shape:output:0.attention__layer/dense/Tensordot/axes:output:09attention__layer/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+attention__layer/dense/Tensordot/GatherV2_1?
&attention__layer/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&attention__layer/dense/Tensordot/Const?
%attention__layer/dense/Tensordot/ProdProd2attention__layer/dense/Tensordot/GatherV2:output:0/attention__layer/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%attention__layer/dense/Tensordot/Prod?
(attention__layer/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(attention__layer/dense/Tensordot/Const_1?
'attention__layer/dense/Tensordot/Prod_1Prod4attention__layer/dense/Tensordot/GatherV2_1:output:01attention__layer/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'attention__layer/dense/Tensordot/Prod_1?
,attention__layer/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,attention__layer/dense/Tensordot/concat/axis?
'attention__layer/dense/Tensordot/concatConcatV2.attention__layer/dense/Tensordot/free:output:0.attention__layer/dense/Tensordot/axes:output:05attention__layer/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'attention__layer/dense/Tensordot/concat?
&attention__layer/dense/Tensordot/stackPack.attention__layer/dense/Tensordot/Prod:output:00attention__layer/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&attention__layer/dense/Tensordot/stack?
*attention__layer/dense/Tensordot/transpose	Transpose attention__layer/concat:output:00attention__layer/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2,
*attention__layer/dense/Tensordot/transpose?
(attention__layer/dense/Tensordot/ReshapeReshape.attention__layer/dense/Tensordot/transpose:y:0/attention__layer/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(attention__layer/dense/Tensordot/Reshape?
'attention__layer/dense/Tensordot/MatMulMatMul1attention__layer/dense/Tensordot/Reshape:output:07attention__layer/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2)
'attention__layer/dense/Tensordot/MatMul?
(attention__layer/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2*
(attention__layer/dense/Tensordot/Const_2?
.attention__layer/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense/Tensordot/concat_1/axis?
)attention__layer/dense/Tensordot/concat_1ConcatV22attention__layer/dense/Tensordot/GatherV2:output:01attention__layer/dense/Tensordot/Const_2:output:07attention__layer/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)attention__layer/dense/Tensordot/concat_1?
 attention__layer/dense/TensordotReshape1attention__layer/dense/Tensordot/MatMul:product:02attention__layer/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2"
 attention__layer/dense/Tensordot?
-attention__layer/dense/BiasAdd/ReadVariableOpReadVariableOp6attention__layer_dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02/
-attention__layer/dense/BiasAdd/ReadVariableOp?
attention__layer/dense/BiasAddBiasAdd)attention__layer/dense/Tensordot:output:05attention__layer/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2 
attention__layer/dense/BiasAdd?
attention__layer/dense/SigmoidSigmoid'attention__layer/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2 
attention__layer/dense/Sigmoid?
1attention__layer/dense_1/Tensordot/ReadVariableOpReadVariableOp:attention__layer_dense_1_tensordot_readvariableop_resource*
_output_shapes

:P(*
dtype023
1attention__layer/dense_1/Tensordot/ReadVariableOp?
'attention__layer/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2)
'attention__layer/dense_1/Tensordot/axes?
'attention__layer/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2)
'attention__layer/dense_1/Tensordot/free?
(attention__layer/dense_1/Tensordot/ShapeShape"attention__layer/dense/Sigmoid:y:0*
T0*
_output_shapes
:2*
(attention__layer/dense_1/Tensordot/Shape?
0attention__layer/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_1/Tensordot/GatherV2/axis?
+attention__layer/dense_1/Tensordot/GatherV2GatherV21attention__layer/dense_1/Tensordot/Shape:output:00attention__layer/dense_1/Tensordot/free:output:09attention__layer/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+attention__layer/dense_1/Tensordot/GatherV2?
2attention__layer/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2attention__layer/dense_1/Tensordot/GatherV2_1/axis?
-attention__layer/dense_1/Tensordot/GatherV2_1GatherV21attention__layer/dense_1/Tensordot/Shape:output:00attention__layer/dense_1/Tensordot/axes:output:0;attention__layer/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2/
-attention__layer/dense_1/Tensordot/GatherV2_1?
(attention__layer/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(attention__layer/dense_1/Tensordot/Const?
'attention__layer/dense_1/Tensordot/ProdProd4attention__layer/dense_1/Tensordot/GatherV2:output:01attention__layer/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2)
'attention__layer/dense_1/Tensordot/Prod?
*attention__layer/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*attention__layer/dense_1/Tensordot/Const_1?
)attention__layer/dense_1/Tensordot/Prod_1Prod6attention__layer/dense_1/Tensordot/GatherV2_1:output:03attention__layer/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2+
)attention__layer/dense_1/Tensordot/Prod_1?
.attention__layer/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense_1/Tensordot/concat/axis?
)attention__layer/dense_1/Tensordot/concatConcatV20attention__layer/dense_1/Tensordot/free:output:00attention__layer/dense_1/Tensordot/axes:output:07attention__layer/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)attention__layer/dense_1/Tensordot/concat?
(attention__layer/dense_1/Tensordot/stackPack0attention__layer/dense_1/Tensordot/Prod:output:02attention__layer/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2*
(attention__layer/dense_1/Tensordot/stack?
,attention__layer/dense_1/Tensordot/transpose	Transpose"attention__layer/dense/Sigmoid:y:02attention__layer/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2.
,attention__layer/dense_1/Tensordot/transpose?
*attention__layer/dense_1/Tensordot/ReshapeReshape0attention__layer/dense_1/Tensordot/transpose:y:01attention__layer/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2,
*attention__layer/dense_1/Tensordot/Reshape?
)attention__layer/dense_1/Tensordot/MatMulMatMul3attention__layer/dense_1/Tensordot/Reshape:output:09attention__layer/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2+
)attention__layer/dense_1/Tensordot/MatMul?
*attention__layer/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2,
*attention__layer/dense_1/Tensordot/Const_2?
0attention__layer/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_1/Tensordot/concat_1/axis?
+attention__layer/dense_1/Tensordot/concat_1ConcatV24attention__layer/dense_1/Tensordot/GatherV2:output:03attention__layer/dense_1/Tensordot/Const_2:output:09attention__layer/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2-
+attention__layer/dense_1/Tensordot/concat_1?
"attention__layer/dense_1/TensordotReshape3attention__layer/dense_1/Tensordot/MatMul:product:04attention__layer/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????(2$
"attention__layer/dense_1/Tensordot?
/attention__layer/dense_1/BiasAdd/ReadVariableOpReadVariableOp8attention__layer_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype021
/attention__layer/dense_1/BiasAdd/ReadVariableOp?
 attention__layer/dense_1/BiasAddBiasAdd+attention__layer/dense_1/Tensordot:output:07attention__layer/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????(2"
 attention__layer/dense_1/BiasAdd?
 attention__layer/dense_1/SigmoidSigmoid)attention__layer/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????(2"
 attention__layer/dense_1/Sigmoid?
1attention__layer/dense_2/Tensordot/ReadVariableOpReadVariableOp:attention__layer_dense_2_tensordot_readvariableop_resource*
_output_shapes

:(*
dtype023
1attention__layer/dense_2/Tensordot/ReadVariableOp?
'attention__layer/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2)
'attention__layer/dense_2/Tensordot/axes?
'attention__layer/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2)
'attention__layer/dense_2/Tensordot/free?
(attention__layer/dense_2/Tensordot/ShapeShape$attention__layer/dense_1/Sigmoid:y:0*
T0*
_output_shapes
:2*
(attention__layer/dense_2/Tensordot/Shape?
0attention__layer/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_2/Tensordot/GatherV2/axis?
+attention__layer/dense_2/Tensordot/GatherV2GatherV21attention__layer/dense_2/Tensordot/Shape:output:00attention__layer/dense_2/Tensordot/free:output:09attention__layer/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+attention__layer/dense_2/Tensordot/GatherV2?
2attention__layer/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2attention__layer/dense_2/Tensordot/GatherV2_1/axis?
-attention__layer/dense_2/Tensordot/GatherV2_1GatherV21attention__layer/dense_2/Tensordot/Shape:output:00attention__layer/dense_2/Tensordot/axes:output:0;attention__layer/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2/
-attention__layer/dense_2/Tensordot/GatherV2_1?
(attention__layer/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(attention__layer/dense_2/Tensordot/Const?
'attention__layer/dense_2/Tensordot/ProdProd4attention__layer/dense_2/Tensordot/GatherV2:output:01attention__layer/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2)
'attention__layer/dense_2/Tensordot/Prod?
*attention__layer/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*attention__layer/dense_2/Tensordot/Const_1?
)attention__layer/dense_2/Tensordot/Prod_1Prod6attention__layer/dense_2/Tensordot/GatherV2_1:output:03attention__layer/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2+
)attention__layer/dense_2/Tensordot/Prod_1?
.attention__layer/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.attention__layer/dense_2/Tensordot/concat/axis?
)attention__layer/dense_2/Tensordot/concatConcatV20attention__layer/dense_2/Tensordot/free:output:00attention__layer/dense_2/Tensordot/axes:output:07attention__layer/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2+
)attention__layer/dense_2/Tensordot/concat?
(attention__layer/dense_2/Tensordot/stackPack0attention__layer/dense_2/Tensordot/Prod:output:02attention__layer/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2*
(attention__layer/dense_2/Tensordot/stack?
,attention__layer/dense_2/Tensordot/transpose	Transpose$attention__layer/dense_1/Sigmoid:y:02attention__layer/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????(2.
,attention__layer/dense_2/Tensordot/transpose?
*attention__layer/dense_2/Tensordot/ReshapeReshape0attention__layer/dense_2/Tensordot/transpose:y:01attention__layer/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2,
*attention__layer/dense_2/Tensordot/Reshape?
)attention__layer/dense_2/Tensordot/MatMulMatMul3attention__layer/dense_2/Tensordot/Reshape:output:09attention__layer/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)attention__layer/dense_2/Tensordot/MatMul?
*attention__layer/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*attention__layer/dense_2/Tensordot/Const_2?
0attention__layer/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0attention__layer/dense_2/Tensordot/concat_1/axis?
+attention__layer/dense_2/Tensordot/concat_1ConcatV24attention__layer/dense_2/Tensordot/GatherV2:output:03attention__layer/dense_2/Tensordot/Const_2:output:09attention__layer/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2-
+attention__layer/dense_2/Tensordot/concat_1?
"attention__layer/dense_2/TensordotReshape3attention__layer/dense_2/Tensordot/MatMul:product:04attention__layer/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2$
"attention__layer/dense_2/Tensordot?
/attention__layer/dense_2/BiasAdd/ReadVariableOpReadVariableOp8attention__layer_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/attention__layer/dense_2/BiasAdd/ReadVariableOp?
 attention__layer/dense_2/BiasAddBiasAdd+attention__layer/dense_2/Tensordot:output:07attention__layer/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2"
 attention__layer/dense_2/BiasAdd?
attention__layer/SqueezeSqueeze)attention__layer/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2
attention__layer/Squeeze?
 attention__layer/ones_like/ShapeShape!attention__layer/Squeeze:output:0*
T0*
_output_shapes
:2"
 attention__layer/ones_like/Shape?
 attention__layer/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 attention__layer/ones_like/Const?
attention__layer/ones_likeFill)attention__layer/ones_like/Shape:output:0)attention__layer/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/ones_likey
attention__layer/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
attention__layer/mul_1/y?
attention__layer/mul_1Mul#attention__layer/ones_like:output:0!attention__layer/mul_1/y:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/mul_1y
attention__layer/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
attention__layer/Equal/y?
attention__layer/EqualEqualCast:y:0!attention__layer/Equal/y:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/Equal?
attention__layer/SelectV2SelectV2attention__layer/Equal:z:0attention__layer/mul_1:z:0!attention__layer/Squeeze:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/SelectV2?
attention__layer/SoftmaxSoftmax"attention__layer/SelectV2:output:0*
T0*'
_output_shapes
:?????????2
attention__layer/Softmax?
attention__layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
attention__layer/ExpandDims/dim?
attention__layer/ExpandDims
ExpandDims"attention__layer/Softmax:softmax:0(attention__layer/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/ExpandDims?
attention__layer/MatMulBatchMatMulV2$attention__layer/ExpandDims:output:0concat/concat:output:0*
T0*+
_output_shapes
:?????????2
attention__layer/MatMul?
attention__layer/Squeeze_1Squeeze attention__layer/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2
attention__layer/Squeeze_1i
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV2#attention__layer/Squeeze_1:output:0concat_1/concat:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????2

concat_2?
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 24
2batch_normalization/moments/mean/reduction_indices?
 batch_normalization/moments/meanMeanconcat_2:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2"
 batch_normalization/moments/mean?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes

:2*
(batch_normalization/moments/StopGradient?
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconcat_2:output:01batch_normalization/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????2/
-batch_normalization/moments/SquaredDifference?
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 28
6batch_normalization/moments/variance/reduction_indices?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2&
$batch_normalization/moments/variance?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1?
)batch_normalization/AssignMovingAvg/decayConst*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/10314*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)batch_normalization/AssignMovingAvg/decay?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_10314*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/10314*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/sub?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/10314*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_10314+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/10314*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOp?
+batch_normalization/AssignMovingAvg_1/decayConst*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/10320*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization/AssignMovingAvg_1/decay?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_10320*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/10320*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/sub?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/10320*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mul?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_10320-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/10320*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp?
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes
:*
dtype02)
'batch_normalization/Cast/ReadVariableOp?
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes
:*
dtype02+
)batch_normalization/Cast_1/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulconcat_2:output:0%batch_normalization/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2?
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????2%
#batch_normalization/batchnorm/add_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAdd?
dense_3/p_re_lu/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/Relu?
dense_3/p_re_lu/ReadVariableOpReadVariableOp'dense_3_p_re_lu_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/p_re_lu/ReadVariableOp
dense_3/p_re_lu/NegNeg&dense_3/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
dense_3/p_re_lu/Neg?
dense_3/p_re_lu/Neg_1Negdense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/Neg_1?
dense_3/p_re_lu/Relu_1Reludense_3/p_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/Relu_1?
dense_3/p_re_lu/mulMuldense_3/p_re_lu/Neg:y:0$dense_3/p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/mul?
dense_3/p_re_lu/addAddV2"dense_3/p_re_lu/Relu:activations:0dense_3/p_re_lu/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_3/p_re_lu/add?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/p_re_lu/add:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_4/BiasAdd?
dense_4/p_re_lu_1/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/Relu?
 dense_4/p_re_lu_1/ReadVariableOpReadVariableOp)dense_4_p_re_lu_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_4/p_re_lu_1/ReadVariableOp?
dense_4/p_re_lu_1/NegNeg(dense_4/p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
dense_4/p_re_lu_1/Neg?
dense_4/p_re_lu_1/Neg_1Negdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/Neg_1?
dense_4/p_re_lu_1/Relu_1Reludense_4/p_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/Relu_1?
dense_4/p_re_lu_1/mulMuldense_4/p_re_lu_1/Neg:y:0&dense_4/p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/mul?
dense_4/p_re_lu_1/addAddV2$dense_4/p_re_lu_1/Relu:activations:0dense_4/p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:??????????2
dense_4/p_re_lu_1/add?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_4/p_re_lu_1/add:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_5/BiasAdd?
dense_5/p_re_lu_2/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/Relu?
 dense_5/p_re_lu_2/ReadVariableOpReadVariableOp)dense_5_p_re_lu_2_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_5/p_re_lu_2/ReadVariableOp?
dense_5/p_re_lu_2/NegNeg(dense_5/p_re_lu_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
dense_5/p_re_lu_2/Neg?
dense_5/p_re_lu_2/Neg_1Negdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/Neg_1?
dense_5/p_re_lu_2/Relu_1Reludense_5/p_re_lu_2/Neg_1:y:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/Relu_1?
dense_5/p_re_lu_2/mulMuldense_5/p_re_lu_2/Neg:y:0&dense_5/p_re_lu_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/mul?
dense_5/p_re_lu_2/addAddV2$dense_5/p_re_lu_2/Relu:activations:0dense_5/p_re_lu_2/mul:z:0*
T0*'
_output_shapes
:?????????@2
dense_5/p_re_lu_2/adds
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMuldense_5/p_re_lu_2/add:z:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mulw
dropout/dropout/ShapeShapedense_5/p_re_lu_2/add:z:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/dropout/Mul_1?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAddi
SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/add?
IdentityIdentitySigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_dense_6_layer_call_and_return_conditional_losses_9785

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_5_layer_call_and_return_conditional_losses_11123

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource%
!p_re_lu_2_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddl
p_re_lu_2/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/Relu?
p_re_lu_2/ReadVariableOpReadVariableOp!p_re_lu_2_readvariableop_resource*
_output_shapes
:@*
dtype02
p_re_lu_2/ReadVariableOpl
p_re_lu_2/NegNeg p_re_lu_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
p_re_lu_2/Negm
p_re_lu_2/Neg_1NegBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/Neg_1s
p_re_lu_2/Relu_1Relup_re_lu_2/Neg_1:y:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/Relu_1?
p_re_lu_2/mulMulp_re_lu_2/Neg:y:0p_re_lu_2/Relu_1:activations:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/mul?
p_re_lu_2/addAddV2p_re_lu_2/Relu:activations:0p_re_lu_2/mul:z:0*
T0*'
_output_shapes
:?????????@2
p_re_lu_2/adde
IdentityIdentityp_re_lu_2/add:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_9762

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_dense_3_layer_call_and_return_conditional_losses_11065

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddi
p_re_lu/ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu/Relu?
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*
_output_shapes	
:?*
dtype02
p_re_lu/ReadVariableOpg
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
p_re_lu/Negj
p_re_lu/Neg_1NegBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu/Neg_1n
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
p_re_lu/Relu_1?
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
p_re_lu/mul?
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*(
_output_shapes
:??????????2
p_re_lu/addd
IdentityIdentityp_re_lu/add:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_dense_4_layer_call_fn_11105

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_96872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
#__inference_signature_wrapper_10168
input_1
input_2
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*%
Tin
2*
Tout
2*'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_91662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?]
?
=__inference_din_layer_call_and_return_conditional_losses_9811
input_1
input_2
input_3
input_4
embedding_9413
attention__layer_9581
attention__layer_9583
attention__layer_9585
attention__layer_9587
attention__layer_9589
attention__layer_9591
batch_normalization_9622
batch_normalization_9624
batch_normalization_9626
batch_normalization_9628
dense_3_9662
dense_3_9664
dense_3_9666
dense_4_9700
dense_4_9702
dense_4_9704
dense_5_9738
dense_5_9740
dense_5_9742
dense_6_9796
dense_6_9798
identity??(attention__layer/StatefulPartitionedCall?+batch_normalization/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinput_3strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceZ

NotEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 2

NotEqual/y
NotEqualNotEqualstrided_slice:output:0NotEqual/y:output:0*
T0*'
_output_shapes
:?????????2

NotEqualc
CastCastNotEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
Cast?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinput_3strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_9413*
Tin
2*
Tout
2*+
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94042#
!embedding/StatefulPartitionedCallq
concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/concat_dim?
concat/concatIdentity*embedding/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2
concat/concat
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinput_4strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_9413*
Tin
2*
Tout
2*'
_output_shapes
:?????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_94382%
#embedding_1/StatefulPartitionedCallu
concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_1/concat_dim?
concat_1/concatIdentity,embedding_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
concat_1/concat?
(attention__layer/StatefulPartitionedCallStatefulPartitionedCallconcat_1/concat:output:0concat/concat:output:0concat/concat:output:0Cast:y:0attention__layer_9581attention__layer_9583attention__layer_9585attention__layer_9587attention__layer_9589attention__layer_9591*
Tin
2
*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

	**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_attention__layer_layer_call_and_return_conditional_losses_95592*
(attention__layer/StatefulPartitionedCalli
concat_2/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_2/axis?
concat_2ConcatV21attention__layer/StatefulPartitionedCall:output:0concat_1/concat:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:?????????2

concat_2?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0batch_normalization_9622batch_normalization_9624batch_normalization_9626batch_normalization_9628*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_92622-
+batch_normalization/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0dense_3_9662dense_3_9664dense_3_9666*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_96492!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_9700dense_4_9702dense_4_9704*
Tin
2*
Tout
2*(
_output_shapes
:??????????*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_96872!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_9738dense_5_9740dense_5_9742*
Tin
2*
Tout
2*'
_output_shapes
:?????????@*%
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_97252!
dense_5/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_97572!
dropout/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_6_9796dense_6_9798*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_97852!
dense_6/StatefulPartitionedCally
SigmoidSigmoid(dense_6/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_9413* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/add?
IdentityIdentitySigmoid:y:0)^attention__layer/StatefulPartitionedCall,^batch_normalization/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::2T
(attention__layer/StatefulPartitionedCall(attention__layer/StatefulPartitionedCall2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
B__inference_dense_4_layer_call_and_return_conditional_losses_11094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddm
p_re_lu_1/ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/Relu?
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
p_re_lu_1/ReadVariableOpm
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
p_re_lu_1/Negn
p_re_lu_1/Neg_1NegBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/Neg_1t
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/Relu_1?
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/mul?
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:??????????2
p_re_lu_1/addf
IdentityIdentityp_re_lu_1/add:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????::::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_11040

inputs,
(embedding_lookup_readvariableop_resource
identity??
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity?
:din/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
??'*
dtype02<
:din/embedding/embeddings/Regularizer/Square/ReadVariableOp?
+din/embedding/embeddings/Regularizer/SquareSquareBdin/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??'2-
+din/embedding/embeddings/Regularizer/Square?
*din/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*din/embedding/embeddings/Regularizer/Const?
(din/embedding/embeddings/Regularizer/SumSum/din/embedding/embeddings/Regularizer/Square:y:03din/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/Sum?
*din/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82,
*din/embedding/embeddings/Regularizer/mul/x?
(din/embedding/embeddings/Regularizer/mulMul3din/embedding/embeddings/Regularizer/mul/x:output:01din/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/mul?
*din/embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*din/embedding/embeddings/Regularizer/add/x?
(din/embedding/embeddings/Regularizer/addAddV23din/embedding/embeddings/Regularizer/add/x:output:0,din/embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(din/embedding/embeddings/Regularizer/addv
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????::K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
3__inference_batch_normalization_layer_call_fn_10919

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_92622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
l
&__inference_p_re_lu_layer_call_fn_9327

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*(
_output_shapes
:??????????*#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_p_re_lu_layer_call_and_return_conditional_losses_93192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :??????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs:

_output_shapes
: 
?|
?
K__inference_attention__layer_layer_call_and_return_conditional_losses_10830
inputs_0
inputs_1
inputs_2
inputs_3+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource-
)dense_2_tensordot_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity?q
Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      2
Tile/multiplesj
TileTileinputs_0Tile/multiples:output:0*
T0*(
_output_shapes
:??????????2
Tiles
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      2
Reshape/shapez
ReshapeReshapeTile:output:0Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapec
subSubReshape:output:0inputs_1*
T0*+
_output_shapes
:?????????2
subc
mulMulReshape:output:0inputs_1*
T0*+
_output_shapes
:?????????2
mule
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat/axis?
concatConcatV2Reshape:output:0inputs_1sub:z:0mul:z:0concat/axis:output:0*
N*
T0*+
_output_shapes
:????????? 2
concat?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: P*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freem
dense/Tensordot/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposeconcat:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:P2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense/BiasAddw
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P2
dense/Sigmoid?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:P(*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes?
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/frees
dense_1/Tensordot/ShapeShapedense/Sigmoid:y:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape?
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2?
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod?
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1?
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack?
dense_1/Tensordot/transpose	Transposedense/Sigmoid:y:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot/transpose?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:(2
dense_1/Tensordot/Const_2?
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????(2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????(2
dense_1/BiasAdd}
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????(2
dense_1/Sigmoid?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:(*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/freeu
dense_2/Tensordot/ShapeShapedense_1/Sigmoid:y:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposedense_1/Sigmoid:y:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????(2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_2/BiasAdd?
SqueezeSqueezedense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims

?????????2	
Squeezeb
ones_like/ShapeShapeSqueeze:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????2
	ones_likeW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_1/ym
mul_1Mulones_like:output:0mul_1/y:output:0*
T0*'
_output_shapes
:?????????2
mul_1W
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2	
Equal/ye
EqualEqualinputs_3Equal/y:output:0*
T0*'
_output_shapes
:?????????2
Equalz
SelectV2SelectV2	Equal:z:0	mul_1:z:0Squeeze:output:0*
T0*'
_output_shapes
:?????????2

SelectV2b
SoftmaxSoftmaxSelectV2:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsSoftmax:softmax:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsv
MatMulBatchMatMulV2ExpandDims:output:0inputs_2*
T0*+
_output_shapes
:?????????2
MatMul{
	Squeeze_1SqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2
	Squeeze_1f
IdentityIdentitySqueeze_1:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*
_input_shapesn
l:?????????:?????????:?????????:?????????:::::::Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
#__inference_din_layer_call_fn_10046
input_1
input_2
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*%
Tin
2*
Tout
2*'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
**
config_proto

CPU

GPU 2J 8*F
fAR?
=__inference_din_layer_call_and_return_conditional_losses_99992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:TP
+
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????
?
input_34
serving_default_input_3:0?????????
;
input_40
serving_default_input_4:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ۘ
?)
dense_feature_columns
sparse_feature_columns
embed_sparse_layers
embed_seq_layers
attention_layer
bn
ffn
dropout
	dense_final

	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?&
_tf_keras_model?&{"class_name": "DIN", "name": "din", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "DIN"}, "training_config": {"loss": "binary_crossentropy", "metrics": [{"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
	att_dense
att_final_dense
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Attention_Layer", "name": "attention__layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"layer was saved without config": true}}
?	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
5
!0
"1
#2"
trackable_list_wrapper
?
$regularization_losses
%	variables
&trainable_variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratem?m?(m?)m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?v?v?(v?)v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?"
	optimizer
(
?0"
trackable_list_wrapper
?
30
41
52
63
74
85
96
7
8
9
10
:11
;12
<13
=14
>15
?16
@17
A18
B19
(20
)21"
trackable_list_wrapper
?
30
41
52
63
74
85
96
7
8
:9
;10
<11
=12
>13
?14
@15
A16
B17
(18
)19"
trackable_list_wrapper
?
Clayer_regularization_losses
Dlayer_metrics
Enon_trainable_variables
Fmetrics
regularization_losses
	variables

Glayers
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
3
embeddings
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 653292, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-05}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
.
L0
M1"
trackable_list_wrapper
?

8kernel
9bias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 40]}}
 "
trackable_list_wrapper
J
40
51
62
73
84
95"
trackable_list_wrapper
J
40
51
62
73
84
95"
trackable_list_wrapper
?
Rlayer_regularization_losses
Slayer_metrics

Tlayers
Umetrics
regularization_losses
	variables
Vnon_trainable_variables
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2din/batch_normalization/gamma
*:(2din/batch_normalization/beta
3:1 (2#din/batch_normalization/moving_mean
7:5 (2'din/batch_normalization/moving_variance
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Wlayer_regularization_losses
Xlayer_metrics

Ylayers
Zmetrics
regularization_losses
	variables
[non_trainable_variables
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	
\
activation

:kernel
;bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?	
a
activation

=kernel
>bias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?	
f
activation

@kernel
Abias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 64, "activation": {"class_name": "PReLU", "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
klayer_regularization_losses
llayer_metrics

mlayers
nmetrics
$regularization_losses
%	variables
onon_trainable_variables
&trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@2din/dense_6/kernel
:2din/dense_6/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
player_regularization_losses
qlayer_metrics

rlayers
smetrics
*regularization_losses
+	variables
tnon_trainable_variables
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*
??'2din/embedding/embeddings
3:1 P2!din/attention__layer/dense/kernel
-:+P2din/attention__layer/dense/bias
5:3P(2#din/attention__layer/dense_1/kernel
/:-(2!din/attention__layer/dense_1/bias
5:3(2#din/attention__layer/dense_2/kernel
/:-2!din/attention__layer/dense_2/bias
%:#	?2din/dense_3/kernel
:?2din/dense_3/bias
(:&?2din/dense_3/p_re_lu/alpha
&:$
??2din/dense_4/kernel
:?2din/dense_4/bias
*:(?2din/dense_4/p_re_lu_1/alpha
%:#	?@2din/dense_5/kernel
:@2din/dense_5/bias
):'@2din/dense_5/p_re_lu_2/alpha
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
X
0
1
2
!3
"4
#5
6
	7"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
?
wlayer_regularization_losses
xlayer_metrics

ylayers
zmetrics
Hregularization_losses
I	variables
{non_trainable_variables
Jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

4kernel
5bias
|regularization_losses
}	variables
~trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 80, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 32]}}
?

6kernel
7bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 80]}}
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
Nregularization_losses
O	variables
?non_trainable_variables
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
L0
M1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	<alpha
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
 "
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
5
:0
;1
<2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
]regularization_losses
^	variables
?non_trainable_variables
_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
	?alpha
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
bregularization_losses
c	variables
?non_trainable_variables
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
	Balpha
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
gregularization_losses
h	variables
?non_trainable_variables
itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?"
?
thresholds
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"?!
_tf_keras_metric?!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
|regularization_losses
}	variables
?non_trainable_variables
~trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
\0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
?0"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
a0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
B0"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
f0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0:.2$Adam/din/batch_normalization/gamma/m
/:-2#Adam/din/batch_normalization/beta/m
):'@2Adam/din/dense_6/kernel/m
#:!2Adam/din/dense_6/bias/m
1:/
??'2Adam/din/embedding/embeddings/m
8:6 P2(Adam/din/attention__layer/dense/kernel/m
2:0P2&Adam/din/attention__layer/dense/bias/m
::8P(2*Adam/din/attention__layer/dense_1/kernel/m
4:2(2(Adam/din/attention__layer/dense_1/bias/m
::8(2*Adam/din/attention__layer/dense_2/kernel/m
4:22(Adam/din/attention__layer/dense_2/bias/m
*:(	?2Adam/din/dense_3/kernel/m
$:"?2Adam/din/dense_3/bias/m
-:+?2 Adam/din/dense_3/p_re_lu/alpha/m
+:)
??2Adam/din/dense_4/kernel/m
$:"?2Adam/din/dense_4/bias/m
/:-?2"Adam/din/dense_4/p_re_lu_1/alpha/m
*:(	?@2Adam/din/dense_5/kernel/m
#:!@2Adam/din/dense_5/bias/m
.:,@2"Adam/din/dense_5/p_re_lu_2/alpha/m
0:.2$Adam/din/batch_normalization/gamma/v
/:-2#Adam/din/batch_normalization/beta/v
):'@2Adam/din/dense_6/kernel/v
#:!2Adam/din/dense_6/bias/v
1:/
??'2Adam/din/embedding/embeddings/v
8:6 P2(Adam/din/attention__layer/dense/kernel/v
2:0P2&Adam/din/attention__layer/dense/bias/v
::8P(2*Adam/din/attention__layer/dense_1/kernel/v
4:2(2(Adam/din/attention__layer/dense_1/bias/v
::8(2*Adam/din/attention__layer/dense_2/kernel/v
4:22(Adam/din/attention__layer/dense_2/bias/v
*:(	?2Adam/din/dense_3/kernel/v
$:"?2Adam/din/dense_3/bias/v
-:+?2 Adam/din/dense_3/p_re_lu/alpha/v
+:)
??2Adam/din/dense_4/kernel/v
$:"?2Adam/din/dense_4/bias/v
/:-?2"Adam/din/dense_4/p_re_lu_1/alpha/v
*:(	?@2Adam/din/dense_5/kernel/v
#:!@2Adam/din/dense_5/bias/v
.:,@2"Adam/din/dense_5/p_re_lu_2/alpha/v
?2?
#__inference_din_layer_call_fn_10669
#__inference_din_layer_call_fn_10721
#__inference_din_layer_call_fn_10046
#__inference_din_layer_call_fn_10098?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__wrapped_model_9166?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
!?
input_1?????????
!?
input_2?????????
%?"
input_3?????????
!?
input_4?????????
?2?
>__inference_din_layer_call_and_return_conditional_losses_10404
>__inference_din_layer_call_and_return_conditional_losses_10617
=__inference_din_layer_call_and_return_conditional_losses_9811
=__inference_din_layer_call_and_return_conditional_losses_9902?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_attention__layer_layer_call_fn_10850?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_attention__layer_layer_call_and_return_conditional_losses_10830?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_batch_normalization_layer_call_fn_10919
3__inference_batch_normalization_layer_call_fn_10932?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10886
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10906?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_10959
'__inference_dropout_layer_call_fn_10954?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_10944
B__inference_dropout_layer_call_and_return_conditional_losses_10949?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_6_layer_call_fn_10978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_6_layer_call_and_return_conditional_losses_10969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_10991?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
KBI
#__inference_signature_wrapper_10168input_1input_2input_3input_4
?2?
)__inference_embedding_layer_call_fn_11023
)__inference_embedding_layer_call_fn_11047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_11016
D__inference_embedding_layer_call_and_return_conditional_losses_11040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_11076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_11065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_4_layer_call_fn_11105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_11094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_5_layer_call_fn_11134?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_11123?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_p_re_lu_layer_call_fn_9327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!???????????????????
?2?
A__inference_p_re_lu_layer_call_and_return_conditional_losses_9319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!???????????????????
?2?
(__inference_p_re_lu_1_layer_call_fn_9348?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!???????????????????
?2?
C__inference_p_re_lu_1_layer_call_and_return_conditional_losses_9340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!???????????????????
?2?
(__inference_p_re_lu_2_layer_call_fn_9369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!???????????????????
?2?
C__inference_p_re_lu_2_layer_call_and_return_conditional_losses_9361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!????????????????????
__inference__wrapped_model_9166?3456789:;<=>?@AB()???
???
???
!?
input_1?????????
!?
input_2?????????
%?"
input_3?????????
!?
input_4?????????
? "3?0
.
output_1"?
output_1??????????
K__inference_attention__layer_layer_call_and_return_conditional_losses_10830?456789???
???
???
"?
inputs/0?????????
&?#
inputs/1?????????
&?#
inputs/2?????????
"?
inputs/3?????????
? "%?"
?
0?????????
? ?
0__inference_attention__layer_layer_call_fn_10850?456789???
???
???
"?
inputs/0?????????
&?#
inputs/1?????????
&?#
inputs/2?????????
"?
inputs/3?????????
? "???????????
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10886b3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_10906b3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
3__inference_batch_normalization_layer_call_fn_10919U3?0
)?&
 ?
inputs?????????
p
? "???????????
3__inference_batch_normalization_layer_call_fn_10932U3?0
)?&
 ?
inputs?????????
p 
? "???????????
B__inference_dense_3_layer_call_and_return_conditional_losses_11065^:;</?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? |
'__inference_dense_3_layer_call_fn_11076Q:;</?,
%?"
 ?
inputs?????????
? "????????????
B__inference_dense_4_layer_call_and_return_conditional_losses_11094_=>?0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
'__inference_dense_4_layer_call_fn_11105R=>?0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_5_layer_call_and_return_conditional_losses_11123^@AB0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? |
'__inference_dense_5_layer_call_fn_11134Q@AB0?-
&?#
!?
inputs??????????
? "??????????@?
B__inference_dense_6_layer_call_and_return_conditional_losses_10969\()/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_6_layer_call_fn_10978O()/?,
%?"
 ?
inputs?????????@
? "???????????
>__inference_din_layer_call_and_return_conditional_losses_10404?3456789:;<=>?@AB()???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
&?#
inputs/2?????????
"?
inputs/3?????????
p
? "%?"
?
0?????????
? ?
>__inference_din_layer_call_and_return_conditional_losses_10617?3456789:;<=>?@AB()???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
&?#
inputs/2?????????
"?
inputs/3?????????
p 
? "%?"
?
0?????????
? ?
=__inference_din_layer_call_and_return_conditional_losses_9811?3456789:;<=>?@AB()???
???
???
!?
input_1?????????
!?
input_2?????????
%?"
input_3?????????
!?
input_4?????????
p
? "%?"
?
0?????????
? ?
=__inference_din_layer_call_and_return_conditional_losses_9902?3456789:;<=>?@AB()???
???
???
!?
input_1?????????
!?
input_2?????????
%?"
input_3?????????
!?
input_4?????????
p 
? "%?"
?
0?????????
? ?
#__inference_din_layer_call_fn_10046?3456789:;<=>?@AB()???
???
???
!?
input_1?????????
!?
input_2?????????
%?"
input_3?????????
!?
input_4?????????
p
? "???????????
#__inference_din_layer_call_fn_10098?3456789:;<=>?@AB()???
???
???
!?
input_1?????????
!?
input_2?????????
%?"
input_3?????????
!?
input_4?????????
p 
? "???????????
#__inference_din_layer_call_fn_10669?3456789:;<=>?@AB()???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
&?#
inputs/2?????????
"?
inputs/3?????????
p
? "???????????
#__inference_din_layer_call_fn_10721?3456789:;<=>?@AB()???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
&?#
inputs/2?????????
"?
inputs/3?????????
p 
? "???????????
B__inference_dropout_layer_call_and_return_conditional_losses_10944\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_10949\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? z
'__inference_dropout_layer_call_fn_10954O3?0
)?&
 ?
inputs?????????@
p
? "??????????@z
'__inference_dropout_layer_call_fn_10959O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
D__inference_embedding_layer_call_and_return_conditional_losses_11016_3/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
D__inference_embedding_layer_call_and_return_conditional_losses_11040W3+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? 
)__inference_embedding_layer_call_fn_11023R3/?,
%?"
 ?
inputs?????????
? "??????????w
)__inference_embedding_layer_call_fn_11047J3+?(
!?
?
inputs?????????
? "??????????:
__inference_loss_fn_0_109913?

? 
? "? ?
C__inference_p_re_lu_1_layer_call_and_return_conditional_losses_9340e?8?5
.?+
)?&
inputs??????????????????
? "&?#
?
0??????????
? ?
(__inference_p_re_lu_1_layer_call_fn_9348X?8?5
.?+
)?&
inputs??????????????????
? "????????????
C__inference_p_re_lu_2_layer_call_and_return_conditional_losses_9361dB8?5
.?+
)?&
inputs??????????????????
? "%?"
?
0?????????@
? ?
(__inference_p_re_lu_2_layer_call_fn_9369WB8?5
.?+
)?&
inputs??????????????????
? "??????????@?
A__inference_p_re_lu_layer_call_and_return_conditional_losses_9319e<8?5
.?+
)?&
inputs??????????????????
? "&?#
?
0??????????
? ?
&__inference_p_re_lu_layer_call_fn_9327X<8?5
.?+
)?&
inputs??????????????????
? "????????????
#__inference_signature_wrapper_10168?3456789:;<=>?@AB()???
? 
???
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????
0
input_3%?"
input_3?????????
,
input_4!?
input_4?????????"3?0
.
output_1"?
output_1?????????