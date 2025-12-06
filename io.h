//---------------------------------------------------------------------------
#ifndef io_h
#define io_h
//---------------------------------------------------------------------------
#include "GISInfo.h"
#include "Sciara.h"
#include "configurationPathLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//---------------------------------------------------------------------------
// Autosave state variables
extern bool storing;		    // if true, automatic saving is enabled
extern int storing_step;    // every `storing_step` steps the configuration is saved
extern char storing_path[]; // path where the configuration is saved
extern struct TGISInfo gis_info_Sz;
extern struct TGISInfo gis_info_generic;
extern struct TGISInfo gis_info_nodata0;
//---------------------------------------------------------------------------

int loadParameters(char const* path, Sciara* sciara);
int saveParameters(char *path, Sciara* sciara);
void printParameters(Sciara* sciara);

int loadMorphology(char* path, Sciara* sciara);
int loadVents(char* path, Sciara* sciara);
int loadEmissionRate(char *path, Sciara* sciara);

int loadAlreadyAllocatedMap(char *path, int* S, int* nS, int lx, int ly);
int loadAlreadyAllocatedMap(char *path, double* S, double* nS, int lx, int ly);

int loadConfiguration(char const *path, Sciara* sciara);
int saveConfiguration(char const *path, Sciara* sciara);

//---------------------------------------------------------------------------
#endif
//---------------------------------------------------------------------------
