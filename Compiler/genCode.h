#ifndef _GENCODE_H_
#define _GENCODE_H_

#include "header.h"

void genLine(int lineNum,char msg[]);
void genInit();
void genFunctStart(SEMTYPE type,char *name,struct param_sem *param_list,int *isMain);
void genFunctEnd(int hasValue);
void genFunctInvoke(char* name,struct SymTable *table,int scope);
void genGlobalVar(SEMTYPE type,char *name);
void genPrintStart();
void genPrintEnd(struct expr_sem *kind);
void genRead(struct expr_sem *expr,int isLocal,int stackNum);
void genGetValue(char *kind,int intValue,float floatValue,double doubleValue,char *name,SEMTYPE type);
void genAssignValue(char *kind,int value,char *name,SEMTYPE type);
void genBooleanLabel(int level,OPERATOR op,int isInt);

#endif

