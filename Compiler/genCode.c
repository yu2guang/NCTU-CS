#include "header.h"
#include "symtab.h"
#include "genCode.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern FILE* fout_code;
extern int genI2F;
extern int genI2F2;

/* tool */

char* Type2Code(SEMTYPE semType){
	char* strBuffer;
	switch(semType){
		case INTEGER_t:
			strBuffer = (char*)malloc(sizeof(char)*1);
			strncpy(strBuffer,"I",1);
			break;
		case FLOAT_t:
			strBuffer = (char*)malloc(sizeof(char)*1);
			strncpy(strBuffer,"F",1);
			break;
		case DOUBLE_t:
			strBuffer = (char*)malloc(sizeof(char)*1);
			strncpy(strBuffer,"F",1);
			break;
		case BOOLEAN_t:
			strBuffer = (char*)malloc(sizeof(char)*1);
			strncpy(strBuffer,"Z",1);
			break;
		case STRING_t:
			strBuffer = (char*)malloc(sizeof(char)*6);
			strncpy(strBuffer,"string",6);
			break;
		case VOID_t:
			strBuffer = (char*)malloc(sizeof(char)*4);
			strncpy(strBuffer,"V",4);
			break;
		case ARRAY_t:
			strBuffer = (char*)malloc(sizeof(char)*5);
			strncpy(strBuffer,"array",5);
			break;
		case FUNCTION_t:
			strBuffer = (char*)malloc(sizeof(char)*8);
			strncpy(strBuffer,"function",8);
			break;
		case VARIABLE_t:
			strBuffer = (char*)malloc(sizeof(char)*8);
			strncpy(strBuffer,"variable",8);
			break;
		case PARAMETER_t:
			strBuffer = (char*)malloc(sizeof(char)*9);
			strncpy(strBuffer,"parameter",9);
			break;
		case CONSTANT_t:
			strBuffer = (char*)malloc(sizeof(char)*8);
			strncpy(strBuffer,"constant",8);
			break;
		case ID_LIST:
			strBuffer = (char*)malloc(sizeof(char)*7);
			strncpy(strBuffer,"id_list",7);
			break;
		case ERROR_t:
			strBuffer = (char*)malloc(sizeof(char)*5);
			strncpy(strBuffer,"error",5);
			break;
		default:
			strBuffer = (char*)malloc(sizeof(char)*7);
			strncpy(strBuffer,"unknown",7);
			break;
	}

	return strBuffer;
}

/* generate code */

void genLine(int lineNum,char msg[]){
	fprintf(fout_code,"; Line #%d: %s",lineNum,msg);
}

void genInit(){
	fprintf(fout_code,".class public output\n");
	fprintf(fout_code,".super java/lang/Object\n");
	fprintf(fout_code,".field public static _sc Ljava/util/Scanner;\n");
}

void genFunctStart(SEMTYPE type,char *name,struct param_sem *param_list,int *isMain){
	fprintf(fout_code,".method public static %s(",name);

	if(strncmp(name,"main",4)==0){ // main
		fprintf(fout_code,"[Ljava/lang/String;)V\n");
		*isMain = 1;
	}
	else{
		if(param_list!=NULL){ // parameter
			while(1){
				fprintf(fout_code,"%s",Type2Code(param_list->pType->type));
				param_list = param_list->next;
				if(param_list==NULL) break;
			}
		}

		fprintf(fout_code,")%s\n",Type2Code(type));
	}

	// limits
	fprintf(fout_code,".limit stack 1000\n");
	fprintf(fout_code,".limit locals 1000\n");

	// scanner
	fprintf(fout_code,"new java/util/Scanner\n");
    fprintf(fout_code,"dup\n");
    fprintf(fout_code,"getstatic java/lang/System/in Ljava/io/InputStream;\n");
    fprintf(fout_code,"invokespecial java/util/Scanner/<init>(Ljava/io/InputStream;)V\n");
    fprintf(fout_code,"putstatic output/_sc Ljava/util/Scanner;\n");
}

void genFunctEnd(int hasValue){ 

	if(hasValue==(-2)){
		fprintf(fout_code,".end method\n");
	}
	else if(hasValue==(-1)){
		fprintf(fout_code,"\treturn\n");
	}
	else{
		fprintf(fout_code,"\tireturn\n");
	}	
}

void genFunctInvoke(char* name,struct SymTable *table,int scope){
	fprintf(fout_code,"\tinvokestatic output/%s(",name);

	struct SymNode *node = lookupSymbol(table,name,scope,__FALSE);

	int paramN = node->attribute->formalParam->paramNum;
	struct PTypeList *paramCur;

	if(paramN!=0){
		paramCur = node->attribute->formalParam->params;

		for(int i=0; i<paramN; i++){		
			fprintf(fout_code,"%s",Type2Code(paramCur->value->type));
			paramCur = paramCur->next;
		}
	}
		
	fprintf(fout_code,")%s\n",Type2Code(node->type->type));
}

void genGlobalVar(SEMTYPE type,char *name){
	fprintf(fout_code,".field public static %s %s\n",name,Type2Code(type));
}

void genPrintStart(){
	fprintf(fout_code,"getstatic java/lang/System/out Ljava/io/PrintStream;\n");   
}

void genPrintEnd(struct expr_sem *kind){

	fprintf(fout_code,"invokevirtual java/io/PrintStream/print(");	

	switch(kind->pType->type){
		case STRING_t:
			fprintf(fout_code,"Ljava/lang/String;");
			break;
		default:
			fprintf(fout_code,"%s",Type2Code(kind->pType->type));
			break;
	}

	fprintf(fout_code,")V\n");	
}

void genRead(struct expr_sem *expr,int isLocal,int stackNum){
	fprintf(fout_code,"\tgetstatic output/_sc Ljava/util/Scanner;\n");
	fprintf(fout_code,"\tinvokevirtual java/util/Scanner/next");
	
	switch(expr->pType->type){
		case INTEGER_t:
			fprintf(fout_code,"Int()I\n");
			if(isLocal!=0){
				fprintf(fout_code,"\tistore %d\n",stackNum);
			}
			else{
				fprintf(fout_code,"\tputstatic output/%s I\n",expr->varRef->id);
			}
			break;
		case FLOAT_t:
			fprintf(fout_code,"Float()F\n");
			if(isLocal!=0){
				fprintf(fout_code,"\tfstore %d\n",stackNum);
			}
			else{
				fprintf(fout_code,"\tputstatic output/%s F\n",expr->varRef->id);
			}
			break;
		case DOUBLE_t:
			fprintf(fout_code,"Float()F\n");
			if(isLocal!=0){
				fprintf(fout_code,"\tfstore %d\n",stackNum);
			}
			else{
				fprintf(fout_code,"\tputstatic output/%s D\n",expr->varRef->id);
			}
			break;
		case BOOLEAN_t:
			fprintf(fout_code,"Boolean()Z\n");
			if(isLocal!=0){
				fprintf(fout_code,"\tistore %d\n",stackNum);
			}
			else{
				fprintf(fout_code,"\tputstatic output/%s Z\n",expr->varRef->id);
			}
			break;
	}
}

void genGetValue(char *kind,int intValue,float floatValue,double doubleValue,char *name,SEMTYPE type){
	if(strncmp(kind,"int",3)==0){
		fprintf(fout_code,"\tldc %d\n",intValue);
		if(genI2F==1){
			fprintf(fout_code,"\ti2f\n");
		}
		else{
			genI2F2 = 1;
		}
	}
	else if(strncmp(kind,"float",5)==0){
		if(genI2F2==1){
			fprintf(fout_code,"\ti2f\n");
		}
		fprintf(fout_code,"\tldc %f\n",floatValue);
	}
	else if(strncmp(kind,"double",6)==0){
		if(genI2F2==1){
			fprintf(fout_code,"\ti2f\n");
		}
		fprintf(fout_code,"\tldc %f\n",doubleValue);
	}
	else if(strncmp(kind,"string",6)==0){
		fprintf(fout_code,"\tldc \"%s\"\n",name);
	}
	else if(strncmp(kind,"bool",4)==0){
		int boolValue;
		if(strncmp(name,"true",4)==0){
			boolValue = 1;
		}
		else{
			boolValue = 0;
		}

		fprintf(fout_code,"\ticonst_%d\n",boolValue);
	}
	else if(strncmp(kind,"local",5)==0){
		if(type==FLOAT_t){
			if(genI2F2==1){
				fprintf(fout_code,"\ti2f\n");
			}
			fprintf(fout_code,"\tfload %d\n",intValue);
		}
		else if(type==DOUBLE_t){
			if(genI2F2==1){
				fprintf(fout_code,"\ti2f\n");
			}
			fprintf(fout_code,"\tfload %d\n",intValue);
		}
		else{
			fprintf(fout_code,"\tiload %d\n",intValue);

			if(genI2F==1){
				fprintf(fout_code,"\ti2f\n");
			}
			else{
				genI2F2 = 1;
			}
		}
	}
	else if(strncmp(kind,"global",6)==0){
		fprintf(fout_code,"\tgetstatic output/%s %s\n",name,Type2Code(type));
	}
}

void genAssignValue(char *kind,int value,char *name,SEMTYPE type){
	if(strncmp(kind,"local",5)==0){
		if(type==DOUBLE_t){
			fprintf(fout_code,"\tfstore %d\n",value);
		}
		else if(type==FLOAT_t){
			fprintf(fout_code,"\tfstore %d\n",value);
		}
		else{
			fprintf(fout_code,"\tistore %d\n",value);
		}			
	}
	else if(strncmp(kind,"global",6)==0){
		fprintf(fout_code,"\tputstatic output/%s %s\n",name,Type2Code(type));
	}
}

void genBooleanLabel(int level,OPERATOR op,int isInt){
	
	if(isInt==1){
		fprintf(fout_code,"\tisub\n");
	}
	else{
		fprintf(fout_code,"\tfcmpl\n");
	}
	
	switch(op){
		case LT_t:
			fprintf(fout_code,"\tiflt ");
			break;
		case LE_t:
			fprintf(fout_code,"\tifle ");
			break;
		case EQ_t:
			fprintf(fout_code,"\tifeq ");
			break;
		case GE_t:
			fprintf(fout_code,"\tifge ");
			break;
		case GT_t:
			fprintf(fout_code,"\tifgt ");
			break;
		case NE_t:
			fprintf(fout_code,"\tifne ");
			break;
	}

	fprintf(fout_code,"Ltrue_%d\n",level);			
	fprintf(fout_code,"\ticonst_0\n");
	fprintf(fout_code,"\tgoto Lnext_%d\n",level);
	fprintf(fout_code,"Ltrue_%d:\n",level);				
	fprintf(fout_code,"\ticonst_1\n");
	fprintf(fout_code,"Lnext_%d:\n",level);	
}