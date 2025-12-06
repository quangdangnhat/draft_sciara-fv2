#ifndef CA_H_
#define CA_H_

#include "GISInfo.h"
#include "vent.h"
#include <math.h>
#include <stdlib.h>

#define VON_NEUMANN_NEIGHBORS 5
#define MOORE_NEIGHBORS 9
#define NUMBER_OF_OUTFLOWS 8

#define MIN_ALG		0
#define PROP_ALG	1

typedef struct
{
	int rows;
	int cols;
} Domain;

// The adopted von Neuman neighborhood // Format: flow_index:cell_label:(row_index,col_index)
//
//   cell_label in [0,1,2,3,4,5,6,7,8]: label assigned to each cell in the neighborhood
//   flow_index in   [0,1,2,3,4,5,6,7]: outgoing flow indices in Mf from cell 0 to the others
//       (row_index,col_index): 2D relative indices of the cells
//
//
//    cells               cells         outflows
//    coordinates         labels        indices
//
//   -1,-1|-1,0| 1,1      |5|1|8|       |4|0|7|
//    0,-1| 0,0| 0,1      |2|0|3|       |1| |2|
//    1,-1| 1,0|-1,1      |6|4|7|       |5|3|6|

typedef struct
{
  int* Xi;
  int* Xj;
} NeighsRelativeCoords;

typedef struct
{
	double* Sz;		    //Altitude
  double* Sz_next;
	double* Sh;	      //Lava thickness
  double* Sh_next;
	double* ST;		    //Lava temperature
  double* ST_next;
	double* Mf;		    //Matrix of the ouflows
	int*    Mv;		    //Matrix of the vents
	bool*   Mb;		    //Matrix of the domain boundaries
	double* Mhs;	    //Matrix of the solidified lava
} Substates;

typedef struct
{
	double Pclock;		//AC clock [s]
	double Pc;				//cell side
	double Pac;				//area of the cell
	double PTsol;			//temperature of solidification
	double PTvent;		//temperature of lava at vent
	double Pr_Tsol;
	double Pr_Tvent;
	double a;				// parameter for computing Pr
	double b;				// parameter for computing Pr
	double Phc_Tsol;
	double Phc_Tvent;
	double c; 				// parameter for computing hc
	double d; 				// parameter for computing hc
	double Pcool;
	double Prho;			//density
	double Pepsilon;	//emissivity
	double Psigma;		//Stephen-Boltzamnn constant
	double Pcv;				//Specific heat
	int algorithm;	
} Parameters;

typedef struct
{
  int    step;
	int    maximum_steps;	//... go for maximum_steps steps (0 for loop)
	double elapsed_time; // elapsed time since simulation start [s]

	unsigned int emission_time;
	vector<TEmissionRate> emission_rate;
	vector<TVent> vent;
	double effusion_duration;
  double total_emitted_lava;

	double stopping_threshold;	// if negative, no pause control is applied
	int    refreshing_step;	// graphic threads are started every repaint_step steps
	double thickness_visual_threshold; 	// in LCMorphology, drawn black only if mMD > visual_threshold
} Simulation;


typedef struct
{
  Domain *domain;
  NeighsRelativeCoords *X;
  Substates *substates;
  Parameters *parameters;
  Simulation *simulation;
} Sciara;

void makeBorder(Sciara *sciara);

// ----------------------------------------------------------------------------
// Memory allocation function for 2D linearized buffers
// ----------------------------------------------------------------------------

void init(Sciara*& sciara);
void simulationInitialize(Sciara* sciara);
void allocateSubstates(Sciara *sciara);
//void deallocateSubstates(Sciara *sciara);
void finalize(Sciara*& sciara);

#endif /* CA_H_ */
