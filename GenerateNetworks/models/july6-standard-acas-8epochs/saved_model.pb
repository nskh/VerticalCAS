��

��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
�
Nadam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameNadam/dense_27/bias/v
{
)Nadam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_27/bias/v*
_output_shapes
:	*
dtype0
�
Nadam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-	*(
shared_nameNadam/dense_27/kernel/v
�
+Nadam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_27/kernel/v*
_output_shapes

:-	*
dtype0
�
Nadam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_26/bias/v
{
)Nadam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_26/bias/v*
_output_shapes
:-*
dtype0
�
Nadam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_26/kernel/v
�
+Nadam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_26/kernel/v*
_output_shapes

:--*
dtype0
�
Nadam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_25/bias/v
{
)Nadam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_25/bias/v*
_output_shapes
:-*
dtype0
�
Nadam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_25/kernel/v
�
+Nadam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_25/kernel/v*
_output_shapes

:--*
dtype0
�
Nadam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_24/bias/v
{
)Nadam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_24/bias/v*
_output_shapes
:-*
dtype0
�
Nadam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_24/kernel/v
�
+Nadam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_24/kernel/v*
_output_shapes

:--*
dtype0
�
Nadam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_23/bias/v
{
)Nadam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_23/bias/v*
_output_shapes
:-*
dtype0
�
Nadam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_23/kernel/v
�
+Nadam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_23/kernel/v*
_output_shapes

:--*
dtype0
�
Nadam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_22/bias/v
{
)Nadam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_22/bias/v*
_output_shapes
:-*
dtype0
�
Nadam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_22/kernel/v
�
+Nadam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_22/kernel/v*
_output_shapes

:--*
dtype0
�
Nadam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_21/bias/v
{
)Nadam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_21/bias/v*
_output_shapes
:-*
dtype0
�
Nadam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*(
shared_nameNadam/dense_21/kernel/v
�
+Nadam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_21/kernel/v*
_output_shapes

:-*
dtype0
�
Nadam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameNadam/dense_27/bias/m
{
)Nadam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_27/bias/m*
_output_shapes
:	*
dtype0
�
Nadam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-	*(
shared_nameNadam/dense_27/kernel/m
�
+Nadam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_27/kernel/m*
_output_shapes

:-	*
dtype0
�
Nadam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_26/bias/m
{
)Nadam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_26/bias/m*
_output_shapes
:-*
dtype0
�
Nadam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_26/kernel/m
�
+Nadam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_26/kernel/m*
_output_shapes

:--*
dtype0
�
Nadam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_25/bias/m
{
)Nadam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_25/bias/m*
_output_shapes
:-*
dtype0
�
Nadam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_25/kernel/m
�
+Nadam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_25/kernel/m*
_output_shapes

:--*
dtype0
�
Nadam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_24/bias/m
{
)Nadam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_24/bias/m*
_output_shapes
:-*
dtype0
�
Nadam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_24/kernel/m
�
+Nadam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_24/kernel/m*
_output_shapes

:--*
dtype0
�
Nadam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_23/bias/m
{
)Nadam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_23/bias/m*
_output_shapes
:-*
dtype0
�
Nadam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_23/kernel/m
�
+Nadam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_23/kernel/m*
_output_shapes

:--*
dtype0
�
Nadam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_22/bias/m
{
)Nadam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_22/bias/m*
_output_shapes
:-*
dtype0
�
Nadam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--*(
shared_nameNadam/dense_22/kernel/m
�
+Nadam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_22/kernel/m*
_output_shapes

:--*
dtype0
�
Nadam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*&
shared_nameNadam/dense_21/bias/m
{
)Nadam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_21/bias/m*
_output_shapes
:-*
dtype0
�
Nadam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-*(
shared_nameNadam/dense_21/kernel/m
�
+Nadam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_21/kernel/m*
_output_shapes

:-*
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:	*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-	* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:-	*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:-*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:--*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:-*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:--*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:-*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:--*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:-*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:--*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:-*
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:--* 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:--*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:-*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:-*
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:-* 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:-*
dtype0
�
serving_default_dense_21_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_21_inputdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_1949809

NoOpNoOp
�[
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�[
value�[B�Z B�Z
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias*
j
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13*
j
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_3* 
6
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 
* 
�
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[momentum_cachem�m�m� m�'m�(m�/m�0m�7m�8m�?m�@m�Gm�Hm�v�v�v� v�'v�(v�/v�0v�7v�8v�?v�@v�Gv�Hv�*

\serving_default* 

0
1*

0
1*
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

ptrace_0* 

qtrace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_25/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_26/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*

G0
H1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_27/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
MG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�}
VARIABLE_VALUENadam/dense_21/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_21/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_22/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_22/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_23/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_23/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_24/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_24/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_25/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_25/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_26/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_26/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_27/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_27/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_21/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_21/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_22/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_22/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_23/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_23/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_24/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_24/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_25/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_25/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_26/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_26/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUENadam/dense_27/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUENadam/dense_27/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Nadam/dense_21/kernel/m/Read/ReadVariableOp)Nadam/dense_21/bias/m/Read/ReadVariableOp+Nadam/dense_22/kernel/m/Read/ReadVariableOp)Nadam/dense_22/bias/m/Read/ReadVariableOp+Nadam/dense_23/kernel/m/Read/ReadVariableOp)Nadam/dense_23/bias/m/Read/ReadVariableOp+Nadam/dense_24/kernel/m/Read/ReadVariableOp)Nadam/dense_24/bias/m/Read/ReadVariableOp+Nadam/dense_25/kernel/m/Read/ReadVariableOp)Nadam/dense_25/bias/m/Read/ReadVariableOp+Nadam/dense_26/kernel/m/Read/ReadVariableOp)Nadam/dense_26/bias/m/Read/ReadVariableOp+Nadam/dense_27/kernel/m/Read/ReadVariableOp)Nadam/dense_27/bias/m/Read/ReadVariableOp+Nadam/dense_21/kernel/v/Read/ReadVariableOp)Nadam/dense_21/bias/v/Read/ReadVariableOp+Nadam/dense_22/kernel/v/Read/ReadVariableOp)Nadam/dense_22/bias/v/Read/ReadVariableOp+Nadam/dense_23/kernel/v/Read/ReadVariableOp)Nadam/dense_23/bias/v/Read/ReadVariableOp+Nadam/dense_24/kernel/v/Read/ReadVariableOp)Nadam/dense_24/bias/v/Read/ReadVariableOp+Nadam/dense_25/kernel/v/Read/ReadVariableOp)Nadam/dense_25/bias/v/Read/ReadVariableOp+Nadam/dense_26/kernel/v/Read/ReadVariableOp)Nadam/dense_26/bias/v/Read/ReadVariableOp+Nadam/dense_27/kernel/v/Read/ReadVariableOp)Nadam/dense_27/bias/v/Read/ReadVariableOpConst*A
Tin:
826	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_1950297
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotal_1count_1totalcountNadam/dense_21/kernel/mNadam/dense_21/bias/mNadam/dense_22/kernel/mNadam/dense_22/bias/mNadam/dense_23/kernel/mNadam/dense_23/bias/mNadam/dense_24/kernel/mNadam/dense_24/bias/mNadam/dense_25/kernel/mNadam/dense_25/bias/mNadam/dense_26/kernel/mNadam/dense_26/bias/mNadam/dense_27/kernel/mNadam/dense_27/bias/mNadam/dense_21/kernel/vNadam/dense_21/bias/vNadam/dense_22/kernel/vNadam/dense_22/bias/vNadam/dense_23/kernel/vNadam/dense_23/bias/vNadam/dense_24/kernel/vNadam/dense_24/bias/vNadam/dense_25/kernel/vNadam/dense_25/bias/vNadam/dense_26/kernel/vNadam/dense_26/bias/vNadam/dense_27/kernel/vNadam/dense_27/bias/v*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1950463��
�

�
E__inference_dense_24_layer_call_and_return_conditional_losses_1950059

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1949809
dense_21_input
unknown:-
	unknown_0:-
	unknown_1:--
	unknown_2:-
	unknown_3:--
	unknown_4:-
	unknown_5:--
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:--

unknown_10:-

unknown_11:-	

unknown_12:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_1949325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�	
�
E__inference_dense_27_layer_call_and_return_conditional_losses_1949444

inputs0
matmul_readvariableop_resource:-	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
*__inference_dense_23_layer_call_fn_1950028

inputs
unknown:--
	unknown_0:-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1949377o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�

�
E__inference_dense_26_layer_call_and_return_conditional_losses_1949428

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�

�
E__inference_dense_25_layer_call_and_return_conditional_losses_1950079

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�	
�
E__inference_dense_27_layer_call_and_return_conditional_losses_1950118

inputs0
matmul_readvariableop_resource:-	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�%
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949768
dense_21_input"
dense_21_1949732:-
dense_21_1949734:-"
dense_22_1949737:--
dense_22_1949739:-"
dense_23_1949742:--
dense_23_1949744:-"
dense_24_1949747:--
dense_24_1949749:-"
dense_25_1949752:--
dense_25_1949754:-"
dense_26_1949757:--
dense_26_1949759:-"
dense_27_1949762:-	
dense_27_1949764:	
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCalldense_21_inputdense_21_1949732dense_21_1949734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1949343�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1949737dense_22_1949739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1949360�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1949742dense_23_1949744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1949377�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_1949747dense_24_1949749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_1949394�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1949752dense_25_1949754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_1949411�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1949757dense_26_1949759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_1949428�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_1949762dense_27_1949764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_1949444x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_1949343

inputs0
matmul_readvariableop_resource:--
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�<
�

I__inference_sequential_3_layer_call_and_return_conditional_losses_1949979

inputs9
'dense_21_matmul_readvariableop_resource:-6
(dense_21_biasadd_readvariableop_resource:-9
'dense_22_matmul_readvariableop_resource:--6
(dense_22_biasadd_readvariableop_resource:-9
'dense_23_matmul_readvariableop_resource:--6
(dense_23_biasadd_readvariableop_resource:-9
'dense_24_matmul_readvariableop_resource:--6
(dense_24_biasadd_readvariableop_resource:-9
'dense_25_matmul_readvariableop_resource:--6
(dense_25_biasadd_readvariableop_resource:-9
'dense_26_matmul_readvariableop_resource:--6
(dense_26_biasadd_readvariableop_resource:-9
'dense_27_matmul_readvariableop_resource:-	6
(dense_27_biasadd_readvariableop_resource:	
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:-*
dtype0{
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_26/MatMulMatMuldense_25/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:-	*
dtype0�
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	h
IdentityIdentitydense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_24_layer_call_and_return_conditional_losses_1949394

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
*__inference_dense_22_layer_call_fn_1950008

inputs
unknown:--
	unknown_0:-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1949360o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�

�
E__inference_dense_22_layer_call_and_return_conditional_losses_1949360

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�

�
E__inference_dense_25_layer_call_and_return_conditional_losses_1949411

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�

�
E__inference_dense_23_layer_call_and_return_conditional_losses_1949377

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
*__inference_dense_27_layer_call_fn_1950108

inputs
unknown:-	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_1949444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�

�
E__inference_dense_26_layer_call_and_return_conditional_losses_1950099

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�f
�
 __inference__traced_save_1950297
file_prefix.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_nadam_dense_21_kernel_m_read_readvariableop4
0savev2_nadam_dense_21_bias_m_read_readvariableop6
2savev2_nadam_dense_22_kernel_m_read_readvariableop4
0savev2_nadam_dense_22_bias_m_read_readvariableop6
2savev2_nadam_dense_23_kernel_m_read_readvariableop4
0savev2_nadam_dense_23_bias_m_read_readvariableop6
2savev2_nadam_dense_24_kernel_m_read_readvariableop4
0savev2_nadam_dense_24_bias_m_read_readvariableop6
2savev2_nadam_dense_25_kernel_m_read_readvariableop4
0savev2_nadam_dense_25_bias_m_read_readvariableop6
2savev2_nadam_dense_26_kernel_m_read_readvariableop4
0savev2_nadam_dense_26_bias_m_read_readvariableop6
2savev2_nadam_dense_27_kernel_m_read_readvariableop4
0savev2_nadam_dense_27_bias_m_read_readvariableop6
2savev2_nadam_dense_21_kernel_v_read_readvariableop4
0savev2_nadam_dense_21_bias_v_read_readvariableop6
2savev2_nadam_dense_22_kernel_v_read_readvariableop4
0savev2_nadam_dense_22_bias_v_read_readvariableop6
2savev2_nadam_dense_23_kernel_v_read_readvariableop4
0savev2_nadam_dense_23_bias_v_read_readvariableop6
2savev2_nadam_dense_24_kernel_v_read_readvariableop4
0savev2_nadam_dense_24_bias_v_read_readvariableop6
2savev2_nadam_dense_25_kernel_v_read_readvariableop4
0savev2_nadam_dense_25_bias_v_read_readvariableop6
2savev2_nadam_dense_26_kernel_v_read_readvariableop4
0savev2_nadam_dense_26_bias_v_read_readvariableop6
2savev2_nadam_dense_27_kernel_v_read_readvariableop4
0savev2_nadam_dense_27_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_nadam_dense_21_kernel_m_read_readvariableop0savev2_nadam_dense_21_bias_m_read_readvariableop2savev2_nadam_dense_22_kernel_m_read_readvariableop0savev2_nadam_dense_22_bias_m_read_readvariableop2savev2_nadam_dense_23_kernel_m_read_readvariableop0savev2_nadam_dense_23_bias_m_read_readvariableop2savev2_nadam_dense_24_kernel_m_read_readvariableop0savev2_nadam_dense_24_bias_m_read_readvariableop2savev2_nadam_dense_25_kernel_m_read_readvariableop0savev2_nadam_dense_25_bias_m_read_readvariableop2savev2_nadam_dense_26_kernel_m_read_readvariableop0savev2_nadam_dense_26_bias_m_read_readvariableop2savev2_nadam_dense_27_kernel_m_read_readvariableop0savev2_nadam_dense_27_bias_m_read_readvariableop2savev2_nadam_dense_21_kernel_v_read_readvariableop0savev2_nadam_dense_21_bias_v_read_readvariableop2savev2_nadam_dense_22_kernel_v_read_readvariableop0savev2_nadam_dense_22_bias_v_read_readvariableop2savev2_nadam_dense_23_kernel_v_read_readvariableop0savev2_nadam_dense_23_bias_v_read_readvariableop2savev2_nadam_dense_24_kernel_v_read_readvariableop0savev2_nadam_dense_24_bias_v_read_readvariableop2savev2_nadam_dense_25_kernel_v_read_readvariableop0savev2_nadam_dense_25_bias_v_read_readvariableop2savev2_nadam_dense_26_kernel_v_read_readvariableop0savev2_nadam_dense_26_bias_v_read_readvariableop2savev2_nadam_dense_27_kernel_v_read_readvariableop0savev2_nadam_dense_27_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *C
dtypes9
725	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :-:-:--:-:--:-:--:-:--:-:--:-:-	:	: : : : : : : : : : :-:-:--:-:--:-:--:-:--:-:--:-:-	:	:-:-:--:-:--:-:--:-:--:-:--:-:-	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:-: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$	 

_output_shapes

:--: 


_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:-	: 

_output_shapes
:	:
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
: :$ 

_output_shapes

:-: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:--: 

_output_shapes
:-:$ 

_output_shapes

:--:  

_output_shapes
:-:$! 

_output_shapes

:--: "

_output_shapes
:-:$# 

_output_shapes

:--: $

_output_shapes
:-:$% 

_output_shapes

:-	: &

_output_shapes
:	:$' 

_output_shapes

:-: (

_output_shapes
:-:$) 

_output_shapes

:--: *

_output_shapes
:-:$+ 

_output_shapes

:--: ,

_output_shapes
:-:$- 

_output_shapes

:--: .

_output_shapes
:-:$/ 

_output_shapes

:--: 0

_output_shapes
:-:$1 

_output_shapes

:--: 2

_output_shapes
:-:$3 

_output_shapes

:-	: 4

_output_shapes
:	:5

_output_shapes
: 
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_1949999

inputs0
matmul_readvariableop_resource:--
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:-*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_22_layer_call_and_return_conditional_losses_1950019

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
.__inference_sequential_3_layer_call_fn_1949690
dense_21_input
unknown:-
	unknown_0:-
	unknown_1:--
	unknown_2:-
	unknown_3:--
	unknown_4:-
	unknown_5:--
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:--

unknown_10:-

unknown_11:-	

unknown_12:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949626o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�%
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949729
dense_21_input"
dense_21_1949693:-
dense_21_1949695:-"
dense_22_1949698:--
dense_22_1949700:-"
dense_23_1949703:--
dense_23_1949705:-"
dense_24_1949708:--
dense_24_1949710:-"
dense_25_1949713:--
dense_25_1949715:-"
dense_26_1949718:--
dense_26_1949720:-"
dense_27_1949723:-	
dense_27_1949725:	
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCalldense_21_inputdense_21_1949693dense_21_1949695*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1949343�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1949698dense_22_1949700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1949360�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1949703dense_23_1949705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1949377�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_1949708dense_24_1949710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_1949394�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1949713dense_25_1949715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_1949411�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1949718dense_26_1949720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_1949428�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_1949723dense_27_1949725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_1949444x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�
�
.__inference_sequential_3_layer_call_fn_1949482
dense_21_input
unknown:-
	unknown_0:-
	unknown_1:--
	unknown_2:-
	unknown_3:--
	unknown_4:-
	unknown_5:--
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:--

unknown_10:-

unknown_11:-	

unknown_12:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�
�
*__inference_dense_26_layer_call_fn_1950088

inputs
unknown:--
	unknown_0:-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_1949428o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
.__inference_sequential_3_layer_call_fn_1949842

inputs
unknown:-
	unknown_0:-
	unknown_1:--
	unknown_2:-
	unknown_3:--
	unknown_4:-
	unknown_5:--
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:--

unknown_10:-

unknown_11:-	

unknown_12:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949451

inputs"
dense_21_1949344:-
dense_21_1949346:-"
dense_22_1949361:--
dense_22_1949363:-"
dense_23_1949378:--
dense_23_1949380:-"
dense_24_1949395:--
dense_24_1949397:-"
dense_25_1949412:--
dense_25_1949414:-"
dense_26_1949429:--
dense_26_1949431:-"
dense_27_1949445:-	
dense_27_1949447:	
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_1949344dense_21_1949346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1949343�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1949361dense_22_1949363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1949360�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1949378dense_23_1949380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1949377�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_1949395dense_24_1949397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_1949394�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1949412dense_25_1949414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_1949411�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1949429dense_26_1949431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_1949428�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_1949445dense_27_1949447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_1949444x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_24_layer_call_fn_1950048

inputs
unknown:--
	unknown_0:-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_1949394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�%
�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949626

inputs"
dense_21_1949590:-
dense_21_1949592:-"
dense_22_1949595:--
dense_22_1949597:-"
dense_23_1949600:--
dense_23_1949602:-"
dense_24_1949605:--
dense_24_1949607:-"
dense_25_1949610:--
dense_25_1949612:-"
dense_26_1949615:--
dense_26_1949617:-"
dense_27_1949620:-	
dense_27_1949622:	
identity�� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall�
 dense_21/StatefulPartitionedCallStatefulPartitionedCallinputsdense_21_1949590dense_21_1949592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1949343�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_1949595dense_22_1949597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1949360�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1949600dense_23_1949602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_1949377�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_1949605dense_24_1949607*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_1949394�
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1949610dense_25_1949612*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_1949411�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1949615dense_26_1949617*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_1949428�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_1949620dense_27_1949622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_1949444x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�K
�
"__inference__wrapped_model_1949325
dense_21_inputF
4sequential_3_dense_21_matmul_readvariableop_resource:-C
5sequential_3_dense_21_biasadd_readvariableop_resource:-F
4sequential_3_dense_22_matmul_readvariableop_resource:--C
5sequential_3_dense_22_biasadd_readvariableop_resource:-F
4sequential_3_dense_23_matmul_readvariableop_resource:--C
5sequential_3_dense_23_biasadd_readvariableop_resource:-F
4sequential_3_dense_24_matmul_readvariableop_resource:--C
5sequential_3_dense_24_biasadd_readvariableop_resource:-F
4sequential_3_dense_25_matmul_readvariableop_resource:--C
5sequential_3_dense_25_biasadd_readvariableop_resource:-F
4sequential_3_dense_26_matmul_readvariableop_resource:--C
5sequential_3_dense_26_biasadd_readvariableop_resource:-F
4sequential_3_dense_27_matmul_readvariableop_resource:-	C
5sequential_3_dense_27_biasadd_readvariableop_resource:	
identity��,sequential_3/dense_21/BiasAdd/ReadVariableOp�+sequential_3/dense_21/MatMul/ReadVariableOp�,sequential_3/dense_22/BiasAdd/ReadVariableOp�+sequential_3/dense_22/MatMul/ReadVariableOp�,sequential_3/dense_23/BiasAdd/ReadVariableOp�+sequential_3/dense_23/MatMul/ReadVariableOp�,sequential_3/dense_24/BiasAdd/ReadVariableOp�+sequential_3/dense_24/MatMul/ReadVariableOp�,sequential_3/dense_25/BiasAdd/ReadVariableOp�+sequential_3/dense_25/MatMul/ReadVariableOp�,sequential_3/dense_26/BiasAdd/ReadVariableOp�+sequential_3/dense_26/MatMul/ReadVariableOp�,sequential_3/dense_27/BiasAdd/ReadVariableOp�+sequential_3/dense_27/MatMul/ReadVariableOp�
+sequential_3/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_21_matmul_readvariableop_resource*
_output_shapes

:-*
dtype0�
sequential_3/dense_21/MatMulMatMuldense_21_input3sequential_3/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
,sequential_3/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_21_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
sequential_3/dense_21/BiasAddBiasAdd&sequential_3/dense_21/MatMul:product:04sequential_3/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-|
sequential_3/dense_21/ReluRelu&sequential_3/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
+sequential_3/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_22_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
sequential_3/dense_22/MatMulMatMul(sequential_3/dense_21/Relu:activations:03sequential_3/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
,sequential_3/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_22_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
sequential_3/dense_22/BiasAddBiasAdd&sequential_3/dense_22/MatMul:product:04sequential_3/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-|
sequential_3/dense_22/ReluRelu&sequential_3/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
+sequential_3/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_23_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
sequential_3/dense_23/MatMulMatMul(sequential_3/dense_22/Relu:activations:03sequential_3/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
,sequential_3/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_23_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
sequential_3/dense_23/BiasAddBiasAdd&sequential_3/dense_23/MatMul:product:04sequential_3/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-|
sequential_3/dense_23/ReluRelu&sequential_3/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
+sequential_3/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_24_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
sequential_3/dense_24/MatMulMatMul(sequential_3/dense_23/Relu:activations:03sequential_3/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
,sequential_3/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_24_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
sequential_3/dense_24/BiasAddBiasAdd&sequential_3/dense_24/MatMul:product:04sequential_3/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-|
sequential_3/dense_24/ReluRelu&sequential_3/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
+sequential_3/dense_25/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_25_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
sequential_3/dense_25/MatMulMatMul(sequential_3/dense_24/Relu:activations:03sequential_3/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
,sequential_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
sequential_3/dense_25/BiasAddBiasAdd&sequential_3/dense_25/MatMul:product:04sequential_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-|
sequential_3/dense_25/ReluRelu&sequential_3/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
+sequential_3/dense_26/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_26_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
sequential_3/dense_26/MatMulMatMul(sequential_3/dense_25/Relu:activations:03sequential_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
,sequential_3/dense_26/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_26_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
sequential_3/dense_26/BiasAddBiasAdd&sequential_3/dense_26/MatMul:product:04sequential_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-|
sequential_3/dense_26/ReluRelu&sequential_3/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
+sequential_3/dense_27/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_27_matmul_readvariableop_resource*
_output_shapes

:-	*
dtype0�
sequential_3/dense_27/MatMulMatMul(sequential_3/dense_26/Relu:activations:03sequential_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
,sequential_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
sequential_3/dense_27/BiasAddBiasAdd&sequential_3/dense_27/MatMul:product:04sequential_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	u
IdentityIdentity&sequential_3/dense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp-^sequential_3/dense_21/BiasAdd/ReadVariableOp,^sequential_3/dense_21/MatMul/ReadVariableOp-^sequential_3/dense_22/BiasAdd/ReadVariableOp,^sequential_3/dense_22/MatMul/ReadVariableOp-^sequential_3/dense_23/BiasAdd/ReadVariableOp,^sequential_3/dense_23/MatMul/ReadVariableOp-^sequential_3/dense_24/BiasAdd/ReadVariableOp,^sequential_3/dense_24/MatMul/ReadVariableOp-^sequential_3/dense_25/BiasAdd/ReadVariableOp,^sequential_3/dense_25/MatMul/ReadVariableOp-^sequential_3/dense_26/BiasAdd/ReadVariableOp,^sequential_3/dense_26/MatMul/ReadVariableOp-^sequential_3/dense_27/BiasAdd/ReadVariableOp,^sequential_3/dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2\
,sequential_3/dense_21/BiasAdd/ReadVariableOp,sequential_3/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_21/MatMul/ReadVariableOp+sequential_3/dense_21/MatMul/ReadVariableOp2\
,sequential_3/dense_22/BiasAdd/ReadVariableOp,sequential_3/dense_22/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_22/MatMul/ReadVariableOp+sequential_3/dense_22/MatMul/ReadVariableOp2\
,sequential_3/dense_23/BiasAdd/ReadVariableOp,sequential_3/dense_23/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_23/MatMul/ReadVariableOp+sequential_3/dense_23/MatMul/ReadVariableOp2\
,sequential_3/dense_24/BiasAdd/ReadVariableOp,sequential_3/dense_24/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_24/MatMul/ReadVariableOp+sequential_3/dense_24/MatMul/ReadVariableOp2\
,sequential_3/dense_25/BiasAdd/ReadVariableOp,sequential_3/dense_25/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_25/MatMul/ReadVariableOp+sequential_3/dense_25/MatMul/ReadVariableOp2\
,sequential_3/dense_26/BiasAdd/ReadVariableOp,sequential_3/dense_26/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_26/MatMul/ReadVariableOp+sequential_3/dense_26/MatMul/ReadVariableOp2\
,sequential_3/dense_27/BiasAdd/ReadVariableOp,sequential_3/dense_27/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_27/MatMul/ReadVariableOp+sequential_3/dense_27/MatMul/ReadVariableOp:W S
'
_output_shapes
:���������
(
_user_specified_namedense_21_input
�
�
.__inference_sequential_3_layer_call_fn_1949875

inputs
unknown:-
	unknown_0:-
	unknown_1:--
	unknown_2:-
	unknown_3:--
	unknown_4:-
	unknown_5:--
	unknown_6:-
	unknown_7:--
	unknown_8:-
	unknown_9:--

unknown_10:-

unknown_11:-	

unknown_12:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949626o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�<
�

I__inference_sequential_3_layer_call_and_return_conditional_losses_1949927

inputs9
'dense_21_matmul_readvariableop_resource:-6
(dense_21_biasadd_readvariableop_resource:-9
'dense_22_matmul_readvariableop_resource:--6
(dense_22_biasadd_readvariableop_resource:-9
'dense_23_matmul_readvariableop_resource:--6
(dense_23_biasadd_readvariableop_resource:-9
'dense_24_matmul_readvariableop_resource:--6
(dense_24_biasadd_readvariableop_resource:-9
'dense_25_matmul_readvariableop_resource:--6
(dense_25_biasadd_readvariableop_resource:-9
'dense_26_matmul_readvariableop_resource:--6
(dense_26_biasadd_readvariableop_resource:-9
'dense_27_matmul_readvariableop_resource:-	6
(dense_27_biasadd_readvariableop_resource:	
identity��dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp�dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:-*
dtype0{
dense_21/MatMulMatMulinputs&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:--*
dtype0�
dense_26/MatMulMatMuldense_25/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:-*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������-�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:-	*
dtype0�
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	h
IdentityIdentitydense_27/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_25_layer_call_fn_1950068

inputs
unknown:--
	unknown_0:-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_1949411o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
��
�
#__inference__traced_restore_1950463
file_prefix2
 assignvariableop_dense_21_kernel:-.
 assignvariableop_1_dense_21_bias:-4
"assignvariableop_2_dense_22_kernel:--.
 assignvariableop_3_dense_22_bias:-4
"assignvariableop_4_dense_23_kernel:--.
 assignvariableop_5_dense_23_bias:-4
"assignvariableop_6_dense_24_kernel:--.
 assignvariableop_7_dense_24_bias:-4
"assignvariableop_8_dense_25_kernel:--.
 assignvariableop_9_dense_25_bias:-5
#assignvariableop_10_dense_26_kernel:--/
!assignvariableop_11_dense_26_bias:-5
#assignvariableop_12_dense_27_kernel:-	/
!assignvariableop_13_dense_27_bias:	(
assignvariableop_14_nadam_iter:	 *
 assignvariableop_15_nadam_beta_1: *
 assignvariableop_16_nadam_beta_2: )
assignvariableop_17_nadam_decay: 1
'assignvariableop_18_nadam_learning_rate: 2
(assignvariableop_19_nadam_momentum_cache: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: =
+assignvariableop_24_nadam_dense_21_kernel_m:-7
)assignvariableop_25_nadam_dense_21_bias_m:-=
+assignvariableop_26_nadam_dense_22_kernel_m:--7
)assignvariableop_27_nadam_dense_22_bias_m:-=
+assignvariableop_28_nadam_dense_23_kernel_m:--7
)assignvariableop_29_nadam_dense_23_bias_m:-=
+assignvariableop_30_nadam_dense_24_kernel_m:--7
)assignvariableop_31_nadam_dense_24_bias_m:-=
+assignvariableop_32_nadam_dense_25_kernel_m:--7
)assignvariableop_33_nadam_dense_25_bias_m:-=
+assignvariableop_34_nadam_dense_26_kernel_m:--7
)assignvariableop_35_nadam_dense_26_bias_m:-=
+assignvariableop_36_nadam_dense_27_kernel_m:-	7
)assignvariableop_37_nadam_dense_27_bias_m:	=
+assignvariableop_38_nadam_dense_21_kernel_v:-7
)assignvariableop_39_nadam_dense_21_bias_v:-=
+assignvariableop_40_nadam_dense_22_kernel_v:--7
)assignvariableop_41_nadam_dense_22_bias_v:-=
+assignvariableop_42_nadam_dense_23_kernel_v:--7
)assignvariableop_43_nadam_dense_23_bias_v:-=
+assignvariableop_44_nadam_dense_24_kernel_v:--7
)assignvariableop_45_nadam_dense_24_bias_v:-=
+assignvariableop_46_nadam_dense_25_kernel_v:--7
)assignvariableop_47_nadam_dense_25_bias_v:-=
+assignvariableop_48_nadam_dense_26_kernel_v:--7
)assignvariableop_49_nadam_dense_26_bias_v:-=
+assignvariableop_50_nadam_dense_27_kernel_v:-	7
)assignvariableop_51_nadam_dense_27_bias_v:	
identity_53��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_22_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_22_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_23_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_23_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_24_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_24_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_25_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_25_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_26_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_26_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_27_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_27_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_nadam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_nadam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp assignvariableop_16_nadam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_nadam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_nadam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_nadam_momentum_cacheIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_nadam_dense_21_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_nadam_dense_21_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp+assignvariableop_26_nadam_dense_22_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_nadam_dense_22_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_dense_23_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_dense_23_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_dense_24_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_dense_24_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_dense_25_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_dense_25_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp+assignvariableop_34_nadam_dense_26_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_nadam_dense_26_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp+assignvariableop_36_nadam_dense_27_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_nadam_dense_27_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp+assignvariableop_38_nadam_dense_21_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_nadam_dense_21_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp+assignvariableop_40_nadam_dense_22_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_nadam_dense_22_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp+assignvariableop_42_nadam_dense_23_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_nadam_dense_23_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp+assignvariableop_44_nadam_dense_24_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_nadam_dense_24_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp+assignvariableop_46_nadam_dense_25_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp)assignvariableop_47_nadam_dense_25_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp+assignvariableop_48_nadam_dense_26_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_nadam_dense_26_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp+assignvariableop_50_nadam_dense_27_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp)assignvariableop_51_nadam_dense_27_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
E__inference_dense_23_layer_call_and_return_conditional_losses_1950039

inputs0
matmul_readvariableop_resource:---
biasadd_readvariableop_resource:-
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:--*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:-*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������-P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������-a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������-w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������-: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������-
 
_user_specified_nameinputs
�
�
*__inference_dense_21_layer_call_fn_1949988

inputs
unknown:-
	unknown_0:-
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1949343o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
dense_21_input7
 serving_default_dense_21_input:0���������<
dense_270
StatefulPartitionedCall:0���������	tensorflow/serving/predict:ͻ
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias"
_tf_keras_layer
�
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13"
trackable_list_wrapper
�
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_32�
.__inference_sequential_3_layer_call_fn_1949482
.__inference_sequential_3_layer_call_fn_1949842
.__inference_sequential_3_layer_call_fn_1949875
.__inference_sequential_3_layer_call_fn_1949690�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0zOtrace_1zPtrace_2zQtrace_3
�
Rtrace_0
Strace_1
Ttrace_2
Utrace_32�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949927
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949979
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949729
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949768�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
�B�
"__inference__wrapped_model_1949325dense_21_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[momentum_cachem�m�m� m�'m�(m�/m�0m�7m�8m�?m�@m�Gm�Hm�v�v�v� v�'v�(v�/v�0v�7v�8v�?v�@v�Gv�Hv�"
	optimizer
,
\serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
btrace_02�
*__inference_dense_21_layer_call_fn_1949988�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
�
ctrace_02�
E__inference_dense_21_layer_call_and_return_conditional_losses_1949999�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0
!:-2dense_21/kernel
:-2dense_21/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
itrace_02�
*__inference_dense_22_layer_call_fn_1950008�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
�
jtrace_02�
E__inference_dense_22_layer_call_and_return_conditional_losses_1950019�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
!:--2dense_22/kernel
:-2dense_22/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_02�
*__inference_dense_23_layer_call_fn_1950028�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
�
qtrace_02�
E__inference_dense_23_layer_call_and_return_conditional_losses_1950039�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0
!:--2dense_23/kernel
:-2dense_23/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
wtrace_02�
*__inference_dense_24_layer_call_fn_1950048�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0
�
xtrace_02�
E__inference_dense_24_layer_call_and_return_conditional_losses_1950059�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zxtrace_0
!:--2dense_24/kernel
:-2dense_24/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
*__inference_dense_25_layer_call_fn_1950068�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0
�
trace_02�
E__inference_dense_25_layer_call_and_return_conditional_losses_1950079�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
!:--2dense_25/kernel
:-2dense_25/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_26_layer_call_fn_1950088�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_26_layer_call_and_return_conditional_losses_1950099�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:--2dense_26/kernel
:-2dense_26/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_27_layer_call_fn_1950108�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_27_layer_call_and_return_conditional_losses_1950118�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:-	2dense_27/kernel
:	2dense_27/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_3_layer_call_fn_1949482dense_21_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_3_layer_call_fn_1949842inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_3_layer_call_fn_1949875inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_3_layer_call_fn_1949690dense_21_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949927inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949979inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949729dense_21_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949768dense_21_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
�B�
%__inference_signature_wrapper_1949809dense_21_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_21_layer_call_fn_1949988inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_21_layer_call_and_return_conditional_losses_1949999inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_22_layer_call_fn_1950008inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_22_layer_call_and_return_conditional_losses_1950019inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_23_layer_call_fn_1950028inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_23_layer_call_and_return_conditional_losses_1950039inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_24_layer_call_fn_1950048inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_24_layer_call_and_return_conditional_losses_1950059inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_25_layer_call_fn_1950068inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_25_layer_call_and_return_conditional_losses_1950079inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_26_layer_call_fn_1950088inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_26_layer_call_and_return_conditional_losses_1950099inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
*__inference_dense_27_layer_call_fn_1950108inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_27_layer_call_and_return_conditional_losses_1950118inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
':%-2Nadam/dense_21/kernel/m
!:-2Nadam/dense_21/bias/m
':%--2Nadam/dense_22/kernel/m
!:-2Nadam/dense_22/bias/m
':%--2Nadam/dense_23/kernel/m
!:-2Nadam/dense_23/bias/m
':%--2Nadam/dense_24/kernel/m
!:-2Nadam/dense_24/bias/m
':%--2Nadam/dense_25/kernel/m
!:-2Nadam/dense_25/bias/m
':%--2Nadam/dense_26/kernel/m
!:-2Nadam/dense_26/bias/m
':%-	2Nadam/dense_27/kernel/m
!:	2Nadam/dense_27/bias/m
':%-2Nadam/dense_21/kernel/v
!:-2Nadam/dense_21/bias/v
':%--2Nadam/dense_22/kernel/v
!:-2Nadam/dense_22/bias/v
':%--2Nadam/dense_23/kernel/v
!:-2Nadam/dense_23/bias/v
':%--2Nadam/dense_24/kernel/v
!:-2Nadam/dense_24/bias/v
':%--2Nadam/dense_25/kernel/v
!:-2Nadam/dense_25/bias/v
':%--2Nadam/dense_26/kernel/v
!:-2Nadam/dense_26/bias/v
':%-	2Nadam/dense_27/kernel/v
!:	2Nadam/dense_27/bias/v�
"__inference__wrapped_model_1949325~ '(/078?@GH7�4
-�*
(�%
dense_21_input���������
� "3�0
.
dense_27"�
dense_27���������	�
E__inference_dense_21_layer_call_and_return_conditional_losses_1949999\/�,
%�"
 �
inputs���������
� "%�"
�
0���������-
� }
*__inference_dense_21_layer_call_fn_1949988O/�,
%�"
 �
inputs���������
� "����������-�
E__inference_dense_22_layer_call_and_return_conditional_losses_1950019\ /�,
%�"
 �
inputs���������-
� "%�"
�
0���������-
� }
*__inference_dense_22_layer_call_fn_1950008O /�,
%�"
 �
inputs���������-
� "����������-�
E__inference_dense_23_layer_call_and_return_conditional_losses_1950039\'(/�,
%�"
 �
inputs���������-
� "%�"
�
0���������-
� }
*__inference_dense_23_layer_call_fn_1950028O'(/�,
%�"
 �
inputs���������-
� "����������-�
E__inference_dense_24_layer_call_and_return_conditional_losses_1950059\/0/�,
%�"
 �
inputs���������-
� "%�"
�
0���������-
� }
*__inference_dense_24_layer_call_fn_1950048O/0/�,
%�"
 �
inputs���������-
� "����������-�
E__inference_dense_25_layer_call_and_return_conditional_losses_1950079\78/�,
%�"
 �
inputs���������-
� "%�"
�
0���������-
� }
*__inference_dense_25_layer_call_fn_1950068O78/�,
%�"
 �
inputs���������-
� "����������-�
E__inference_dense_26_layer_call_and_return_conditional_losses_1950099\?@/�,
%�"
 �
inputs���������-
� "%�"
�
0���������-
� }
*__inference_dense_26_layer_call_fn_1950088O?@/�,
%�"
 �
inputs���������-
� "����������-�
E__inference_dense_27_layer_call_and_return_conditional_losses_1950118\GH/�,
%�"
 �
inputs���������-
� "%�"
�
0���������	
� }
*__inference_dense_27_layer_call_fn_1950108OGH/�,
%�"
 �
inputs���������-
� "����������	�
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949729x '(/078?@GH?�<
5�2
(�%
dense_21_input���������
p 

 
� "%�"
�
0���������	
� �
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949768x '(/078?@GH?�<
5�2
(�%
dense_21_input���������
p

 
� "%�"
�
0���������	
� �
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949927p '(/078?@GH7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������	
� �
I__inference_sequential_3_layer_call_and_return_conditional_losses_1949979p '(/078?@GH7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������	
� �
.__inference_sequential_3_layer_call_fn_1949482k '(/078?@GH?�<
5�2
(�%
dense_21_input���������
p 

 
� "����������	�
.__inference_sequential_3_layer_call_fn_1949690k '(/078?@GH?�<
5�2
(�%
dense_21_input���������
p

 
� "����������	�
.__inference_sequential_3_layer_call_fn_1949842c '(/078?@GH7�4
-�*
 �
inputs���������
p 

 
� "����������	�
.__inference_sequential_3_layer_call_fn_1949875c '(/078?@GH7�4
-�*
 �
inputs���������
p

 
� "����������	�
%__inference_signature_wrapper_1949809� '(/078?@GHI�F
� 
?�<
:
dense_21_input(�%
dense_21_input���������"3�0
.
dense_27"�
dense_27���������	