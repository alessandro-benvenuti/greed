#define GUROBI
#include "src/env/ged_env.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

using NodeLabel = int;
//using EdgeLabel = ged::NoLabel;
using EdgeLabel = int;
using GeometricNodeLabel = std::vector<double>;

class SEDEditCosts: public ged::EditCosts<NodeLabel, EdgeLabel>
{
public:
	double node_ins_cost_fun(const NodeLabel& node_label) const
	{
		return 0;
	}
	
	double node_del_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_rel_cost_fun(const NodeLabel& node_label_1, const NodeLabel& node_label_2) const
	{
		return node_label_1 != node_label_2;
	}
	
	double edge_ins_cost_fun(const EdgeLabel& edge_label) const
	{
		return 0;
	}
	
	double edge_del_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_rel_cost_fun(const EdgeLabel& edge_label_1, const EdgeLabel& edge_label_2) const
	{
		return 0;
	}
};

class GEDEditCosts: public ged::EditCosts<NodeLabel, EdgeLabel>
{
public:
	double node_ins_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_del_cost_fun(const NodeLabel& node_label) const
	{
		return 1;
	}
	
	double node_rel_cost_fun(const NodeLabel& node_label_1, const NodeLabel& node_label_2) const
	{
		return node_label_1 != node_label_2;
	}
	
	double edge_ins_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_del_cost_fun(const EdgeLabel& edge_label) const
	{
		return 1;
	}
	
	double edge_rel_cost_fun(const EdgeLabel& edge_label_1, const EdgeLabel& edge_label_2) const
	{
		return 0;
	}
};

class GeometricEditCosts: public ged::EditCosts<GeometricNodeLabel, EdgeLabel>
{
public:
	GeometricEditCosts(double coordinate_scale, double node_cost, double edge_cost):
		coordinate_scale_(coordinate_scale), node_cost_(node_cost), edge_cost_(edge_cost)
	{
		if (!std::isfinite(coordinate_scale_) || coordinate_scale_ <= 0) {
			throw std::invalid_argument("coordinate_scale must be finite and positive");
		}
	}
	double node_ins_cost_fun(const GeometricNodeLabel&) const { return node_cost_; }
	double node_del_cost_fun(const GeometricNodeLabel&) const { return node_cost_; }
	double node_rel_cost_fun(const GeometricNodeLabel& a, const GeometricNodeLabel& b) const
	{
		if (a.size() != 3 || b.size() != 3) throw std::invalid_argument("3D node labels must have length 3");
		double squared = 0;
		for (std::size_t i = 0; i < 3; ++i) {
			if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::invalid_argument("non-finite coordinate");
			squared += (a[i] - b[i]) * (a[i] - b[i]);
		}
		return std::sqrt(squared) / coordinate_scale_;
	}
	double edge_ins_cost_fun(const EdgeLabel&) const { return edge_cost_; }
	double edge_del_cost_fun(const EdgeLabel&) const { return edge_cost_; }
	double edge_rel_cost_fun(const EdgeLabel&, const EdgeLabel&) const { return 0; }
private:
	double coordinate_scale_, node_cost_, edge_cost_;
};

ged::Options::GEDMethod method_name_to_option(std::string name)
{
	if (name == "anchor_aware_ged") {
		return ged::Options::GEDMethod::ANCHOR_AWARE_GED;
	} else if (name == "blp_no_edge_labels") {
		return ged::Options::GEDMethod::BLP_NO_EDGE_LABELS;
	} else if (name == "branch") {
		return ged::Options::GEDMethod::BRANCH;
	} else if (name == "f2") {
		return ged::Options::GEDMethod::F2;
	} else if (name == "ipfp") {
		return ged::Options::GEDMethod::IPFP;
	} else {
		throw std::invalid_argument("unknown method");
	}
}

using Data = std::pair< std::vector< NodeLabel >, std::vector< std::pair< int, int >>>;
using GeometricData = std::pair< std::vector< GeometricNodeLabel >, std::vector< std::pair< int, int >>>;

template<class Env, class GraphId, class DataType>
void add_geometric_graph(Env& env, GraphId graph_id, const DataType& graph)
{
	for (int i = 0; i < static_cast<int>(graph.first.size()); ++i) {
		if (graph.first[i].size() != 3) throw std::invalid_argument("every node coordinate must have length 3");
		for (double value: graph.first[i]) if (!std::isfinite(value)) throw std::invalid_argument("non-finite coordinate");
		env.add_node(graph_id, i, graph.first[i]);
	}
	std::set<std::pair<int,int>> seen;
	for (auto edge: graph.second) {
		if (edge.first < 0 || edge.second < 0 || edge.first >= static_cast<int>(graph.first.size()) || edge.second >= static_cast<int>(graph.first.size()))
			throw std::invalid_argument("invalid edge endpoint");
		if (edge.first == edge.second) continue;
		if (edge.second < edge.first) std::swap(edge.first, edge.second);
		if (seen.insert(edge).second) env.add_edge(graph_id, edge.first, edge.second, 0);
	}
}

std::tuple<double, double> geometric_ged(const GeometricData& g, const GeometricData& h,
	std::string method, std::string method_args, double coordinate_scale,
	double node_cost, double edge_cost)
{
	ged::GEDEnv<int, GeometricNodeLabel, EdgeLabel> env;
	auto gi = env.add_graph();
	auto hi = env.add_graph();
	add_geometric_graph(env, gi, g);
	add_geometric_graph(env, hi, h);
	env.set_edit_costs(new GeometricEditCosts(coordinate_scale, node_cost, edge_cost));
	env.init();
	// GEDLIB's MIP base defaults to anchoring node zero. Vascular node indices
	// carry no semantics, so F2 must allow every permutation.
	if (method == "f2" && method_args.find("--map-root-to-root") == std::string::npos) {
		method_args += " --map-root-to-root FALSE";
	}
	env.set_method(method_name_to_option(method), method_args);
	env.init_method();
	env.run_method(gi, hi);
	return std::make_tuple(env.get_lower_bound(gi, hi), env.get_upper_bound(gi, hi));
}

std::string solver_version()
{
	return std::to_string(GRB_VERSION_MAJOR) + "." + std::to_string(GRB_VERSION_MINOR) + "." + std::to_string(GRB_VERSION_TECHNICAL);
}

std::tuple< double, double >
sed(const Data& g, const Data& h, std::vector< std::string > method_name, std::vector< std::string > method_args)
{
	ged::GEDEnv< int, NodeLabel, EdgeLabel > env;
	
	auto gi = env.add_graph();
	const auto& g_x = g.first;
	const auto& g_edge_index = g.second;
	for (int i = 0; i < (int)g_x.size(); ++i) {
		env.add_node(gi, i, g_x[i]);
	}
	for (const auto& p: g_edge_index) {
		//env.add_edge(gi, p.first, p.second, ged::NoLabel());
		env.add_edge(gi, p.first, p.second, 0);
	}
	
	auto hi = env.add_graph();
	const auto& h_x = h.first;
	const auto& h_edge_index = h.second;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		env.add_node(hi, i, h_x[i]);
	}
	for (const auto& p: h_edge_index) {
		//env.add_edge(hi, p.first, p.second, ged::NoLabel());
		env.add_edge(hi, p.first, p.second, 0);
	}
	
	// quick-fix: remove
	if (method_name[0] == "ged_f2") {
		env.set_edit_costs(new GEDEditCosts());
		method_name[0] = "f2";
	} else if (method_name[0] == "ged_branch") {
		env.set_edit_costs(new GEDEditCosts());
		method_name[0] = "branch";
	} else {
		env.set_edit_costs(new SEDEditCosts());
	}
	
	env.init();
	double lb, ub;
	if (method_name.size() == 1) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	} else if (method_name.size() == 2) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		env.set_method(method_name_to_option(method_name[1]), method_args[1]);
		env.init_method();
		env.run_method(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	}
	
	return std::make_tuple(lb, ub);
}

std::tuple< double, double, std::vector< int >, std::vector< int >>
sed_plus(const Data& g, const Data& h,
		std::vector< std::string > method_name, std::vector< std::string > method_args)
{
	ged::GEDEnv< int, NodeLabel, EdgeLabel > env;
	
	auto gi = env.add_graph();
	const auto& g_x = g.first;
	const auto& g_edge_index = g.second;
	for (int i = 0; i < (int)g_x.size(); ++i) {
		env.add_node(gi, i, g_x[i]);
	}
	for (const auto& p: g_edge_index) {
		//env.add_edge(gi, p.first, p.second, ged::NoLabel());
		env.add_edge(gi, p.first, p.second, 0);
	}
	
	auto hi = env.add_graph();
	const auto& h_x = h.first;
	const auto& h_edge_index = h.second;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		env.add_node(hi, i, h_x[i]);
	}
	for (const auto& p: h_edge_index) {
		//env.add_edge(hi, p.first, p.second, ged::NoLabel());
		env.add_edge(hi, p.first, p.second, 0);
	}
	
	env.set_edit_costs(new SEDEditCosts());
	env.init();
	double lb, ub;
	if (method_name.size() == 1) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	} else if (method_name.size() == 2) {
		env.set_method(method_name_to_option(method_name[0]), method_args[0]);
		env.init_method();
		env.run_method(gi, hi);
		lb = env.get_lower_bound(gi, hi);
		env.set_method(method_name_to_option(method_name[1]), method_args[1]);
		env.init_method();
		env.run_method(gi, hi);
		ub = env.get_upper_bound(gi, hi);
	}
	
	ged::NodeMap node_map = env.get_node_map(gi, hi);
	
	std::vector< int > node_mask_idx;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		if (node_map.pre_image(i) != ged::GEDGraph::dummy_node()) {
			node_mask_idx.push_back(i);
		}
	}
	
	std::vector< std::vector< bool >> g_adj(g_x.size(), std::vector< bool >(g_x.size(), false));
	for (const auto &p: g_edge_index) {
		g_adj[p.first][p.second] = true;
	}
	
	std::vector< int > edge_mask_idx;
	for (int i = 0; i < (int)h_edge_index.size(); ++i) {
		const auto &p = h_edge_index[i];
		if (
				(node_map.pre_image(p.first) != ged::GEDGraph::dummy_node()) &&
				(node_map.pre_image(p.second) != ged::GEDGraph::dummy_node()) &&
				g_adj[node_map.pre_image(p.first)][node_map.pre_image(p.second)]
			) {
			edge_mask_idx.push_back(i);
		}
	}
	
// 	std::vector< std::pair< std::size_t, std::size_t >> rel;
// 	node_map.as_relation(rel);
// 	std::cout << std::endl;
// 	for (const auto &p: rel) {
// 		std::cout << p.first << " -> " << p.second << std::endl;
// 	}
// 	std::cout << std::endl;
	
	return std::make_tuple(lb, ub, node_mask_idx, edge_mask_idx);
}

std::pair< std::vector< std::pair< std::size_t, std::size_t >>, double >
sed_align(const Data& g, const Data& h)
{
	ged::GEDEnv< int, NodeLabel, EdgeLabel > env;
	auto gi = env.add_graph();
	auto& g_x = g.first;
	auto& g_edge_index = g.second;
	for (int i = 0; i < (int)g_x.size(); ++i) {
		env.add_node(gi, i, g_x[i]);
	}
	for (auto& p: g_edge_index) {
		//env.add_edge(gi, p.first, p.second, ged::NoLabel());
		env.add_edge(gi, p.first, p.second, 0);
	}
	auto hi = env.add_graph();
	auto& h_x = h.first;
	auto& h_edge_index = h.second;
	for (int i = 0; i < (int)h_x.size(); ++i) {
		env.add_node(hi, i, h_x[i]);
	}
	for (auto& p: h_edge_index) {
		//env.add_edge(hi, p.first, p.second, ged::NoLabel());
		env.add_edge(hi, p.first, p.second, 0);
	}
	env.set_edit_costs(new SEDEditCosts());
	env.init();
	env.set_method(ged::Options::GEDMethod::BRANCH);
	env.init_method();
	env.run_method(gi, hi);
	ged::NodeMap node_map = env.get_node_map(gi, hi);
	std::vector< std::pair< std::size_t, std::size_t >> rel;
	node_map.as_relation(rel);
	return std::make_pair(rel, node_map.induced_cost());
}

PYBIND11_MODULE(pyged, m) {
	m.def("sed", &sed);
	m.def("sed_plus", &sed_plus);
	m.def("sed_align", &sed_align);
	m.def("geometric_ged", &geometric_ged, pybind11::arg("g"), pybind11::arg("h"),
		pybind11::arg("method") = "f2", pybind11::arg("method_args") = "--threads 1",
		pybind11::arg("coordinate_scale") = 1.0, pybind11::arg("node_cost") = 1.0,
		pybind11::arg("edge_cost") = 1.0);
	m.def("solver_version", &solver_version);
	m.def("gedlib_version", []() { return std::string("v1.0+gurobi-bound-patch"); });
}
