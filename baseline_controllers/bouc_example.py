from ast import Num
import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# imports for trieste active learning of feasibility region (https://secondmind-labs.github.io/trieste/4.3.0/notebooks/feasible_sets.html)
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
import tensorflow as tf
from trieste.space import Box
import trieste
from trieste.data import Dataset
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
import scipy.stats as stats

class Sim_constrained:
    threshold = 2.0
    
    @staticmethod
    def objective(x):
        f = np.exp(-x)
        return f
    @staticmethod
    def constraint(x):
        c = pow(x, 2)
        return c

OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"

def observer_constrained(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_constrained.objective(query_points)),
            CONSTRAINT: Dataset(query_points, Sim_constrained.constraint(query_points)),
        }
        
class Sim_unconstrained:
    @staticmethod
    def objective(x):
        # make threshold penalty a list of nulls the same length as x
        threshold_penalty = np.zeros((len(x),1))
        for idx in range(len(x.numpy().flatten())):
            if pow(x[idx,0],2) > 2.0:
                threshold_penalty[idx,0] = pow(x[idx,0],2)

        f = np.exp(-x.numpy())
        return_value = np.array(f + threshold_penalty).reshape(-1,1)
        return return_value

def observer_unconstrained(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_unconstrained.objective(query_points)),
        }

def create_bo_model(data):
    gpr = build_gpr(data, search_space)
    return GaussianProcessRegression(gpr)

# set the random seed
rand_seed = 42
np.random.seed(rand_seed)

search_space = Box([1.0], [2.0])
num_init_points = 4
num_steps = 4
init_data_cons = observer_constrained(search_space.sample(num_init_points))
init_query_points = init_data_cons[OBJECTIVE].query_points
init_data_uncons = observer_unconstrained(init_query_points)
init_model_cons = trieste.utils.map_values(create_bo_model, init_data_cons) 
init_model_uncons = trieste.utils.map_values(create_bo_model, init_data_uncons)




# architecture diagram for the constrained optimization
# create a figure 
fig = plt.figure(figsize=(10,6))
# create an axis in the bottom left corner of the figure which is about 15% the weight and height of the figure
c_of_x_ax = fig.add_axes([0.2, 0.1, 0.15, 0.15])
# plot the constraint function c(x) = x^2
x = np.linspace(1, 2, 500).reshape(-1,1)
y = Sim_constrained.constraint(tf.convert_to_tensor(x))
c_of_x_ax.plot(x, y, color='black', label='c(x)', alpha=0.5)
# plot the model mean and uncertainty of the model's estimate of the constraint
const_mean, const_variance = init_model_cons['CONSTRAINT'].predict(x)
c_of_x_ax.plot(x, const_mean, color='black',linestyle="--", label='Model', alpha=0.5)
c_of_x_ax.fill_between(x.flatten(), const_mean.numpy().flatten() - np.sqrt(const_variance.numpy().flatten()), const_mean.numpy().flatten() + np.sqrt(const_variance.numpy().flatten()), color='black', alpha=0.2)
c_of_x_ax.set_title("$c(x)$", fontsize='large')
c_of_x_ax.set_xlabel("$x$", fontsize='large')
# put a red line at y = 2.0 to indicate the threshold
c_of_x_ax.axhline(y=Sim_constrained.threshold, color='red', linestyle='--', label='Threshold')
#c_of_x_ax.legend(loc='upper left', fontsize='small')

# just to the right of that plot, connected by an arrow, plot the probability of feasibility. not as a heatmap but as a line plot
p_of_x_ax = fig.add_axes([0.45, 0.1, 0.15, 0.15])
# compute probability of feasibility
p_feasible = stats.norm.cdf((Sim_constrained.threshold - const_mean.numpy().flatten()) / np.sqrt(const_variance.numpy().flatten()))
# plot the probability of feasibility as a line plot
p_of_x_ax.plot(x, p_feasible, color='black', label='p(x)', alpha=0.5)
# plot the threshold as a red line
p_of_x_ax.set_title("$Pr(c(x) \leq 2)$", fontsize='large')
p_of_x_ax.set_xlabel("$x$", fontsize='large')
# add a small arrow connecting the c_of_x_ax to the p_of_x_ax
p_of_x_ax.annotate("", xy=(-0.3, 0.5), xytext=(-0.6, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# plot f(x) and the model's estimate of f(x) just above where c(x) is plotted
f_of_x_ax = fig.add_axes([0.2, 0.4, 0.15, 0.15])
# plot the objective function f(x) = exp(-x)
f_of_x_ax.plot(x, Sim_constrained.objective(tf.convert_to_tensor(x)), color='black', label='f(x)', alpha=0.5)
# plot the model mean and uncertainty of the model's estimate of f(x)
f_mean, f_variance = init_model_cons['OBJECTIVE'].predict(x)
f_of_x_ax.plot(x, f_mean, color='black',linestyle="--", label='Model', alpha=0.5)
f_of_x_ax.fill_between(x.flatten(), f_mean.numpy().flatten() - np.sqrt(f_variance.numpy().flatten()), f_mean.numpy().flatten() + np.sqrt(f_variance.numpy().flatten()), color='black', alpha=0.2)
f_of_x_ax.set_title("$f(x)$", fontsize='large')
# no xlabel needed for this row

# now, just overtop of p_of_x_ax, plot the expected improvement as a lineplot. this is the shaded region as in f(x) minus the best performance so far
# find the minimum cost observed thus far
min_cost = np.min(init_data_cons[OBJECTIVE].observations.flatten())
# compute the expected improvement as the difference between the model mean and the minimum cost observed thus far
ei_of_x_ax = fig.add_axes([0.45, 0.4, 0.15, 0.15])
# expected improvement is int_-infty^+infty (min_cost - y)_+ * p(y|x) dy
# where y=f(x) is the model's estimate of the objective function
# and p(y|x) is the predictive marginal density of the objective function y = f(x)
# compute expected improvement using that integral
ei = np.zeros((x.shape[0],))
for i in range(x.shape[0]):
    for z_value in np.arange(-4, 4, 0.1):
        # likelihood of this z_value
        p_z = stats.norm.pdf(z_value)
        # compute the expected improvement for this z_value
        f_predicted_z = f_mean[i] + z_value * np.sqrt(f_variance[i])  # f(x) + z_value * std
        # compute the expected improvement
        ei_this_z = (max(0, min_cost - f_predicted_z))
        ei[i] += ei_this_z * p_z * 0.1  # integrate over z_value using the trapezoidal rule




ei_of_x_ax.plot(x, ei, color='black',linestyle="-", alpha=0.5)
ei_of_x_ax.set_title("$EI(x)$", fontsize='large')
# no xlabel needed for this row
ei_of_x_ax.annotate("", xy=(-0.3, 0.5), xytext=(-0.6, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# now element-wise multiply the expected improvement by the probability of feasibility to get the expected constrained improvement
eci = ei.flatten() * p_feasible.flatten()
eci_of_x_ax = fig.add_axes([0.7, 0.25, 0.15, 0.15])
eci_of_x_ax.plot(x, eci, color='black',linestyle="-", label='Model', alpha=0.5)
# put a star at the maximum of the expected constrained improvement
max_idx = np.argmax(eci)
eci_of_x_ax.plot(x[max_idx], eci[max_idx], 'ro', markersize=8, markerfacecolor='none',label='Max ECI', alpha=0.5)
eci_of_x_ax.set_title("$ECI(x)$", fontsize='large')
eci_of_x_ax.set_xlabel("$x$", fontsize='large')

eci_of_x_ax.annotate("", xy=(-0.35, 0.4), xytext=(-0.6, 0.1), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="->", color='black', lw=1.5))
eci_of_x_ax.annotate("", xy=(-0.35, 0.6), xytext=(-0.6, 0.9), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

# save figure as svg called "arch_plots.svg"
plt.savefig("arch_plots.svg", dpi=450)
plt.savefig("arch_plots.png", dpi=450)




fig, ax = plt.subplots(2,2,figsize=(10,6))
ax[0,0].set_title("Unconstrained",fontsize='x-large')
ax[0,1].set_title("Constrained",fontsize='x-large')
# indicate number of init points in init_string
init_string = "Initial\nn = " + str(num_init_points)
ax[0,0].set_ylabel(init_string,fontsize='x-large',rotation=0,labelpad=25)
final_string = "Final\nn = " + str(num_init_points + num_steps)
ax[1,0].set_ylabel(final_string,fontsize='x-large',rotation=0,labelpad=25)

# annotate f'(x) in math script on ax[0,0], bottom right corner
ax[0,0].annotate(r"$f'(x)$", xy=(0.1, 0.85), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax[1,0].annotate(r"$f'(x)$", xy=(0.1, 0.85), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# annotate f(x) in math script on ax[1,0], bottom left
ax[0,1].annotate(r"$f(x)$", xy=(0.1, 0.2), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax[1,1].annotate(r"$f(x)$", xy=(0.1, 0.2), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')


# annotate c(x) at (0.8, 0.5) in figure coordinates (not axis)
#ax[0,1].annotate(r"$c(x)$", xy=(0.8, -0.1), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# get rid of xticks on ax[0,0] and ax[0,1]
ax[0,0].set_xticks([])
ax[0,1].set_xticks([])

# set xlabel for ax[1,0] and ax[1,1] as "x" in math script
ax[1,0].set_xlabel(r"$x$",fontsize='xx-large')
ax[1,1].set_xlabel(r"$x$",fontsize='xx-large')

# for both the unconstrained plots, plot the true objective function
x = np.linspace(1, 2, 500).reshape(-1,1)
y = Sim_unconstrained.objective(tf.convert_to_tensor(x))
# plot it as a solid black line
ax[0,0].plot(x, y, color='black', label='True Objective')
ax[1,0].plot(x, y, color='black', label='True Objective')
y = Sim_constrained.objective(tf.convert_to_tensor(x))
# plot it as a solid black line
ax[0,1].plot(x, y, color='black', label='True Objective')
ax[1,1].plot(x, y, color='black', label='True Objective')

# plot the query points and observations as black dots
ax[0,0].scatter(init_query_points, init_data_uncons[OBJECTIVE].observations, color='black', label='Observations', alpha=0.5)
ax[0,1].scatter(init_query_points, init_data_cons[OBJECTIVE].observations, color='black', label='Observations', alpha=0.5)
# plot the model mean and uncertainty as a shaded area
ax[0,0].plot(x, init_model_uncons['OBJECTIVE'].predict(x)[0], color='black',linestyle="--", label='Model', alpha=0.5)
ax[0,0].fill_between(x.flatten(), init_model_uncons['OBJECTIVE'].predict(x)[0].numpy().flatten() - np.sqrt(init_model_uncons['OBJECTIVE'].predict(x)[1].numpy().flatten()), init_model_uncons['OBJECTIVE'].predict(x)[0].numpy().flatten() + np.sqrt(init_model_uncons['OBJECTIVE'].predict(x)[1].numpy().flatten()), color='black', alpha=0.2)

ax[0,1].plot(x, init_model_cons['OBJECTIVE'].predict(x)[0], color='black',linestyle="--", label='Model', alpha=0.5)
ax[0,1].fill_between(x.flatten(), init_model_cons['OBJECTIVE'].predict(x)[0].numpy().flatten() - np.sqrt(init_model_cons['OBJECTIVE'].predict(x)[1].numpy().flatten()), init_model_cons['OBJECTIVE'].predict(x)[0].numpy().flatten() + np.sqrt(init_model_cons['OBJECTIVE'].predict(x)[1].numpy().flatten()), color='black', alpha=0.2)

const_mean, const_variance = init_model_cons['CONSTRAINT'].predict(x)
# compute probability of feasibility
p_feasible = stats.norm.cdf((Sim_constrained.threshold - const_mean.numpy().flatten()) / np.sqrt(const_variance.numpy().flatten()))

# plot the feasibility as the background color on the plot. x coordinates with high probabilities of feasibility should be green with low probabilities red
# heatmap
X,Y = np.meshgrid(x.flatten(), np.linspace(ax[0,1].get_ylim()[0], ax[0,1].get_ylim()[1], 500))
Z = np.tile(p_feasible, (500,1))
mesh = ax[0,1].pcolormesh(X, Y, Z, shading='auto', cmap='RdBu', alpha=0.5,vmin=0.0,vmax=1.0)
# add a colorbar with labels "certainly safe" and "certainly unsafe" at 1 and 0 respectively
#cbar = plt.colorbar(ax[0,1].collections[0], ax=ax[0,1],cmap='RdBu')
cbar = fig.colorbar(mesh,location='bottom')
cbar.set_ticks([0,0.5, 1])
cbar.set_ticklabels(["Certainly\nInfeasible","Inferred\nThreshold of $c(x)$", "Certainly\nFeasible"])

ax[0,0].legend(loc = 'lower right',fontsize='large')


#plt.show()


pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_constrained.threshold)
eci = trieste.acquisition.ExpectedConstrainedImprovement(
    OBJECTIVE, pof.using(CONSTRAINT)
)
cons_rule = EfficientGlobalOptimization(eci)  


cons_bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_constrained, search_space)
uncons_bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_unconstrained, search_space)

cons_bo_result = cons_bo.optimize(
    num_steps,
    init_data_cons,
    init_model_cons,
    cons_rule
)
uncons_bo_result = uncons_bo.optimize(
    num_steps,
    init_data_uncons,
    init_model_uncons
)

cons_data = cons_bo_result.try_get_final_datasets()
uncons_data = uncons_bo_result.try_get_final_dataset()
cons_model = cons_bo_result.try_get_final_models()
uncons_model = uncons_bo_result.try_get_final_model()


# plot the final results for the unconstrained model
ax[1,0].scatter(uncons_data.query_points.numpy().flatten(), uncons_data.observations.numpy().flatten(), color='black', label='Observations', alpha=0.5)
ax[1,0].plot(x, uncons_model.predict(x)[0], color='black',linestyle="--", label='Model', alpha=0.5)
ax[1,0].fill_between(x.flatten(), uncons_model.predict(x)[0].numpy().flatten() - np.sqrt(uncons_model.predict(x)[1].numpy().flatten()), uncons_model.predict(x)[0].numpy().flatten() + np.sqrt(uncons_model.predict(x)[1].numpy().flatten()), color='black', alpha=0.2)

# do the same for the constrained model
ax[1,1].scatter(cons_data[OBJECTIVE].query_points.numpy().flatten(), cons_data[OBJECTIVE].observations.numpy().flatten(), color='black', label='Observations', alpha=0.5)
ax[1,1].plot(x, cons_model['OBJECTIVE'].predict(x)[0], color='black',linestyle="--", label='Model', alpha=0.5)
ax[1,1].fill_between(x.flatten(), cons_model['OBJECTIVE'].predict(x)[0].numpy().flatten() - np.sqrt(cons_model['OBJECTIVE'].predict(x)[1].numpy().flatten()), cons_model['OBJECTIVE'].predict(x)[0].numpy().flatten() + np.sqrt(cons_model['OBJECTIVE'].predict(x)[1].numpy().flatten()), color='black', alpha=0.2)

const_mean, const_variance = cons_model['CONSTRAINT'].predict(x)
# compute probability of feasibility
p_feasible = stats.norm.cdf((Sim_constrained.threshold - const_mean.numpy().flatten()) / np.sqrt(const_variance.numpy().flatten()))

# plot the feasibility as the background color on the plot. x coordinates with high probabilities of feasibility should be green with low probabilities red
# heatmap
X,Y = np.meshgrid(x.flatten(), np.linspace(ax[1,1].get_ylim()[0], ax[1,1].get_ylim()[1], 500))
Z = np.tile(p_feasible, (500,1))
mesh2 = ax[1,1].pcolormesh(X, Y, Z, shading='auto', cmap='RdBu', alpha=0.5,vmin=0.0,vmax=1.0)

plt.tight_layout()

plt.savefig("bouc_example.png", dpi=450)
#plt.savefig("bouc_example.svg", dpi=450)
plt.show()

