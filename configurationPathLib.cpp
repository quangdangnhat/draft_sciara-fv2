#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "configurationPathLib.h"


char * strrev (char * string)
{
  char *start = string;
  char *left = string;
  char ch;

  while (*string++) /* find end of string */
    ;
  string -= 2;

  while (left < string)
  {
    ch = *left;
    *left++ = *string;
    *string-- = ch;
  }

  return(start);
}

void ConfigurationIdPath(char config_file_path[], char config_id_str[])
{
  int pos = 0;

  for (int i=strlen(config_file_path); i>=0; i--)
    //if ( config_file_path[i] == '/' || config_file_path[i] == '\\')
    if ( config_file_path[i] == '_' )
    {
      pos = i;
      break;
    }

  if (pos == 0)
    strcpy(config_id_str, "\0");
  else
  {
    strcpy(config_id_str, config_file_path);
    config_id_str[pos] = '\0';
  }
}

void ConfigurationFilePath(char config_file_path[], char const * name, char const *suffix, char file_path[])
{
    /*
      Builds the full file path in `file_path` for the file to open:
      `config_file_path` is the full path of the configuration file. e.g.: curti_000000000000.cfg
      `name` is the name of the substate to open. e.g.: Morphology
      `suffix` is the file extension to open. e.g.: .asc

    */
    strcpy(file_path, "\0");            // initialize file_path to the empty string
    strcat(file_path, config_file_path); // file_path is initialized to the full configuration file path
    int lp = strlen(file_path)-4;       // length of the string without the .cfg extension
    file_path[lp] = '\0';               // file_path = file_path without the .cfg extension
    strcat(file_path, "_");             // file_path = file_path + '_' e.g.: curti_000000000000_
    strcat(file_path, name);            // file_path = file_path + name e.g.: curti_000000000000_Morphology
    strcat(file_path, suffix);          // file_path = file_path + .asc e.g.: curti_000000000000_Morphology.asc
}
//---------------------------------------------------------------------------
int GetStepFromConfigurationFile(char config_file_path[])
{
  char step_str[150] = "\0";

  strcpy(step_str, config_file_path);
  step_str[strlen(step_str)-4] = '\0';
  strcpy(step_str, strrev(step_str));
  step_str[12] = '\0';
  strcpy(step_str, strrev(step_str));
  return atoi(step_str);
}
//---------------------------------------------------------------------------
bool ConfigurationFileSavingPath(char config_file_path[], int step, char const * name, char const * suffix, char file_path[])
{
  char p[32];                         // string containing the step
  char ps[] = "000000000000";         // 12-digit string to contain the calculation step (step)

  strcpy(file_path, "\0");            // initialize file_path to the empty string
  strcat(file_path, config_file_path); // file_path is initialized to the configuration file path. e.g.: "c:\\simulazioni\\curti"
  if (step >= 0) {
    strcat(file_path, "_");           // file_path = file_path + "_". e.g.: "c:\\simulazioni\\curti_"
    sprintf(p, "%d", step);		      // convert AC step to string. e.g.: p = "345"
    int lp = strlen (p);              // compute length of string p
    int lps = strlen (ps);            // compute length of string ps
    if (lps < lp)                     // check
      return false;
    ps[lps-lp] = '\0';                // shorten ps so it can contain p
    strcat(ps, p);                    // ps = ps + p. e.g.: ps = "000000000345"
    strcat(file_path, ps);            // file_path = file_path + ps. e.g.: "c:\\simulazioni\\curti_000000000345"

  }
  if (strcmp(name, ""))               // if name is not the empty string
  {
    strcat(file_path, "_");		        // file_path = file_path + "_". e.g.: "c:\\simulazioni\\curti_000000000345_"
    strcat(file_path, name);          // file_path = file_path + name. e.g.: "c:\\simulazioni\\curti_000000000345_Morphology"
  }

  strcat(file_path, suffix);          //file_path = file_path + ."asc". Es: "c:\\simulazioni\\curti_000000000345_Morphology.stt"

  return true;
}
//---------------------------------------------------------------------------
