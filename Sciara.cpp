#include "Sciara.h"
#include "cal2DBuffer.h"

void allocateSubstates(Sciara *sciara)
{
    int size = sciara->domain->rows * sciara->domain->cols;

    sciara->substates->Sz = new double[size]();
    sciara->substates->Sz_next = new double[size]();
    sciara->substates->Sh = new double[size]();
    sciara->substates->Sh_next = new double[size]();
    sciara->substates->ST = new double[size]();
    sciara->substates->ST_next = new double[size]();
    sciara->substates->Mf = new double[size * NUMBER_OF_OUTFLOWS]();
    sciara->substates->Mb = new bool[size]();
    sciara->substates->Mhs = new double[size]();
}

void deallocateSubstates(Sciara *sciara)
{
    delete[] sciara->substates->Sz;
    delete[] sciara->substates->Sz_next;
    delete[] sciara->substates->Sh;
    delete[] sciara->substates->Sh_next;
    delete[] sciara->substates->ST;
    delete[] sciara->substates->ST_next;
    delete[] sciara->substates->Mf;
    delete[] sciara->substates->Mb;
    delete[] sciara->substates->Mhs;
}

void evaluatePowerLawParams(double PTvent, double PTsol, double value_sol, double value_vent, double &k1, double &k2)
{
	k2 = ( log10(value_vent) - log10(value_sol) ) / (PTvent - PTsol) ;
	k1 = log10(value_sol) - k2*(PTsol);
}

void simulationInitialize(Sciara* sciara)
{
  // declarations
  unsigned int maximum_number_of_emissions = 0;

  // reset the AC step
  sciara->simulation->step = 0;
  sciara->simulation->elapsed_time = 0;

  // determine maximum number of steps
  for (unsigned int i = 0; i < sciara->simulation->emission_rate.size(); i++)
    if (maximum_number_of_emissions < sciara->simulation->emission_rate[i].size())
      maximum_number_of_emissions = sciara->simulation->emission_rate[i].size();
  //maximum_steps_from_emissions = (int)(emission_time/Pclock*maximum_number_of_emissions);
  sciara->simulation->effusion_duration = sciara->simulation->emission_time * maximum_number_of_emissions;
  sciara->simulation->total_emitted_lava = 0;

  // define the morphology border
  makeBorder(sciara);

  // compute a, b (viscosity parameters) and c, d (shear-resistance parameters)
  evaluatePowerLawParams(
      sciara->parameters->PTvent,
      sciara->parameters->PTsol,
      sciara->parameters->Pr_Tsol,
      sciara->parameters->Pr_Tvent,
      sciara->parameters->a,
      sciara->parameters->b);
  evaluatePowerLawParams(
      sciara->parameters->PTvent,
      sciara->parameters->PTsol,
      sciara->parameters->Phc_Tsol,
      sciara->parameters->Phc_Tvent,
      sciara->parameters->c,
      sciara->parameters->d);
}

int _Xi[] = {0, -1,  0,  0,  1, -1,  1,  1, -1}; // Xj: Moore neighborhood row coordinates (see below)
int _Xj[] = {0,  0, -1,  1,  0, -1, -1,  1,  1}; // Xj: Moore neighborhood col coordinates (see below)
void init(Sciara*& sciara)
{
  sciara = new Sciara;
  sciara->domain = new Domain;

  sciara->X = new NeighsRelativeCoords;
  sciara->X->Xi = new int[MOORE_NEIGHBORS];
  sciara->X->Xj = new int[MOORE_NEIGHBORS];
  for (int n=0; n<MOORE_NEIGHBORS; n++)
  {
    sciara->X->Xi[n] = _Xi[n];
    sciara->X->Xj[n] = _Xj[n];
  }

  sciara->substates = new Substates; //Substates allocation is done when the confiugration is loaded
  sciara->parameters = new Parameters;
  sciara->simulation = new Simulation;
}

void finalize(Sciara*& sciara)
{
  deallocateSubstates(sciara);
  delete sciara->domain;
  delete sciara->X->Xi;
  delete sciara->X->Xj;
  delete sciara->X;
  delete sciara->substates;
  delete sciara->parameters;
  delete sciara->simulation;
  delete sciara;
  sciara = NULL;
}

void makeBorder(Sciara *sciara)
{
	int j, i;

  // first row
	i = 0;
	for (j = 0; j < sciara->domain->cols; j++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // last row
	i = sciara->domain->rows - 1;
	for (j = 0; j < sciara->domain->cols; j++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // first column
	j = 0;
	for (i = 0; i < sciara->domain->rows; i++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // last column
	j = sciara->domain->cols - 1;
	for (i = 0; i < sciara->domain->rows; i++)
		if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0)
			calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);

  // the rest
	for (int i = 1; i < sciara->domain->rows - 1; i++)
		for (int j = 1; j < sciara->domain->cols - 1; j++)
			if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i, j) >= 0) {
				for (int k = 1; k < MOORE_NEIGHBORS; k++)
					if (calGetMatrixElement(sciara->substates->Sz, sciara->domain->cols, i+sciara->X->Xi[k], j+sciara->X->Xj[k]) < 0)
          {
		      calSetMatrixElement(sciara->substates->Mb, sciara->domain->cols, i, j, true);
						break;
					}
			}
}