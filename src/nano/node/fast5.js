// Ian Holmes, 2017
// A slimmed-down port of David Matei's fast5.hpp which can be found here:
// https://github.com/mateidavid/fast5/blob/master/src/fast5.hpp

var extend = require('extend')

var hdf5_module = require('hdf5')
var hdf5_globals = require('hdf5/lib/globals')

var hdf5 = hdf5_module.hdf5
var h5lt = hdf5_module.h5lt
var h5tb = hdf5_module.h5tb

var Access = hdf5_globals.Access
var H5OType = hdf5_globals.H5OType

// Constructor

var File = function (filename) {
  extend (this,
          { channel_id_params: {},
            raw_samples_read_names: [],
            event_detection_groups: [],
            event_detection_read_names: {},
            basecall_groups: [],
            basecall_group_descriptions: {},
            basecall_strand_groups: [[], [], []] })
  
  if (filename)
    this.open (filename)
}

// HDF5 helpers

File.prototype.path_parent_child = function (path) {
  var path_split = path.split('/').slice(1), child = path_split.pop(), parent = '/' + path_split.join('/')
  return { parent: parent.length ? parent : '/',
           child: child }
}

File.prototype.get_object_type = function (path) {
  var pc = this.path_parent_child (path)
  if (pc.parent.length && pc.parent !== '/' && !this.group_exists(pc.parent))
    return H5OType.H5O_TYPE_UNKNOWN
  var parent = pc.parent === '/' ? this.file : this.file.openGroup(pc.parent)
  var isChild = {}
  parent.getMemberNames().forEach (function (child) { isChild[child] = true })
  if (!isChild[pc.child])
    return H5OType.H5O_TYPE_UNKNOWN
  return parent.getChildType (pc.child)
}

File.prototype.group_exists = function (path) {
  return this.get_object_type(path) === H5OType.H5O_TYPE_GROUP
}

File.prototype.dataset_exists = function (path) {
  return this.get_object_type(path) === H5OType.H5O_TYPE_DATASET
}

File.prototype.attribute_exists = function (path) {
  var pc = this.path_parent_child (path)
  var attrs = this.file.getDatasetAttributes(pc.parent)
  return attrs.hasOwnProperty (pc.child)
}

File.prototype.get_attr = function (path) {
  var pc = this.path_parent_child (path)
  return this.file.getDatasetAttributes(pc.parent)[pc.child]
}

File.prototype.get_dataset = function (path) {
  return h5lt.readDataset (this.file.id, path)
}

File.prototype.table_to_object = function (table) {
  var obj = {}
  table.forEach (function (col) { obj[col.name] = col })
  return obj
}

// Fast5 internal paths

File.prototype.file_version_path = function() { return "/file_version"; }
File.prototype.channel_id_path = function()   { return "/UniqueGlobalKey/channel_id"; }
File.prototype.tracking_id_path = function()  { return "/UniqueGlobalKey/tracking_id"; }
File.prototype.sequences_path = function()    { return "/Sequences/Meta"; }
File.prototype.raw_samples_root_path = function() { return "/Raw/Reads"; }
File.prototype.raw_samples_params_path = function(rn)
{
  return this.raw_samples_root_path() + "/" + rn;
}
File.prototype.raw_samples_path = function(rn)
{
  return this.raw_samples_root_path() + "/" + rn + "/Signal";
}
File.prototype.event_detection_root_path = function() { return "/Analyses"; }
File.prototype.event_detection_group_prefix = function() { return "EventDetection_"; }
File.prototype.event_detection_group_path = function (gr)
{
  return this.event_detection_root_path() + "/" + this.event_detection_group_prefix() + gr;
}
File.prototype.event_detection_events_params_path = function (gr, rn)
{
  return this.event_detection_group_path(gr) + "/Reads/" + rn;
}
File.prototype.event_detection_events_path = function (gr, rn)
{
  return this.event_detection_group_path(gr) + "/Reads/" + rn + "/Events";
}
File.prototype.basecall_root_path = function() { return "/Analyses"; }
File.prototype.basecall_group_prefix = function() { return "Basecall_"; }
var _strand_name = [ "template", "complement", "2D" ]
File.prototype.strand_name = function(st)
{
  return _strand_name[st];
}
File.prototype.basecall_strand_subgroup = function(st)
{
  return "BaseCalled_" + this.strand_name(st);
}
File.prototype.basecall_group_path = function (gr)
{
  return this.basecall_root_path() + "/" + this.basecall_group_prefix() + gr;
}
File.prototype.basecall_strand_group_path = function (gr, st)
{
  return this.basecall_group_path(gr) + "/" + this.basecall_strand_subgroup(st);
}
File.prototype.basecall_log_path = function (gr)
{
  return this.basecall_group_path(gr) + "/Log";
}
File.prototype.basecall_fastq_path = function (gr, st)
{
  return this.basecall_strand_group_path(gr, st) + "/Fastq";
}
File.prototype.basecall_events_path = function (gr, st)
{
  return this.basecall_strand_group_path(gr, st) + "/Events";
}
File.prototype.basecall_config_path = function(gr)
{
  return this.basecall_group_path(gr) + "/Configuration";
}
File.prototype.basecall_summary_path = function(gr)
{
  return this.basecall_group_path(gr) + "/Summary";
}

// Cache updaters

File.prototype.open = function (filename) {
  this.file = new hdf5.File (filename, Access.ACC_RDONLY)
  this.reload()
}

File.prototype.reload = function() {
  this.load_channel_id_params()
  this.load_raw_samples_read_names()
  this.load_event_detection_groups()
  this.load_basecall_groups()
}

File.prototype.load_channel_id_params = function() {
  this.channel_id_params = this.file.getDatasetAttributes (this.channel_id_path())
}

File.prototype.load_raw_samples_read_names = function() {
  if (this.group_exists(this.raw_samples_root_path())) {
    var group = this.file.openGroup (this.raw_samples_root_path())
    this.raw_samples_read_names = group.getMemberNamesByCreationOrder()
  }
}

File.prototype.load_event_detection_groups = function() {
  var fast5 = this
  if (this.group_exists(this.event_detection_root_path())) {
    var ed_gr_prefix = this.event_detection_group_prefix()
    var group = this.file.openGroup (this.event_detection_root_path())
    var ed_group_names = group.getMemberNamesByCreationOrder()
    this.event_detection_groups = ed_group_names.filter (function (ed_group_name) {
      return ed_group_name.substr(0,ed_gr_prefix.length) === ed_gr_prefix
    }).map (function (ed_group_name) {
      return ed_group_name.substr (ed_gr_prefix.length)
    })
    this.event_detection_groups.forEach (function (ed_gr_suffix) {
      var ed_group = group.openGroup (ed_gr_prefix + ed_gr_suffix + '/Reads')
      fast5.event_detection_read_names[ed_gr_suffix] = ed_group.getMemberNamesByCreationOrder()
    })
  }
}

File.prototype.load_basecall_groups = function() {
  var fast5 = this
  if (this.group_exists (this.basecall_root_path())) {
    var bc_gr_prefix = this.basecall_group_prefix()
    var group = this.file.openGroup (this.basecall_root_path())
    var bc_group_names = group.getMemberNamesByCreationOrder()
    this.basecall_groups = bc_group_names.filter (function (bc_group_name) {
      return bc_group_name.substr(0,bc_gr_prefix.length) === bc_gr_prefix
    }).map (function (bc_group_name) {
      return bc_group_name.substr (bc_gr_prefix.length)
    })
    this.basecall_groups.forEach (function (gr) {
      var desc = { name: "?",
                   version: "?",
                   have_subgroup: [false, false, false],
                   have_fastq: [false, false, false],
                   have_events: [false, false, false] }
      var attr = fast5.file.getDatasetAttributes (fast5.basecall_group_path(gr))
      if (attr.name === "ONT Sequencing Workflow") {
        desc.name = "metrichor";
        desc.version = (attr["chimaera version"] || "?") + "+" +
          (attr["dragonet version"] || "?");
      } else if (attr.name === "MinKNOW-Live-Basecalling") {
        desc.name = "minknow";
        desc.version = attr.version || "?"
      } else if (attr.name == "ONT Albacore Sequencing Software") {
        desc.name = "albacore";
        desc.version = attr.version || "?"
      }
      for (var st = 0; st < 3; ++st) {
        if (fast5.group_exists (fast5.basecall_strand_group_path(gr, st))) {
          desc.have_subgroup[st] = true
          fast5.basecall_strand_groups[st].push(gr)
          desc.have_fastq[st] = fast5.dataset_exists (fast5.basecall_fastq_path(gr, st))
          desc.have_events[st] = fast5.dataset_exists (fast5.basecall_events_path(gr, st))
          if (st === 0)
            desc.ed_gr = fast5.detect_basecall_event_detection_group(gr)
        }
      }
      if (desc.have_subgroup[0] || desc.have_subgroup[1])
        desc.bc_1d_gr = gr
      else if (desc.have_subgroup[2])
        desc.bc_1d_gr = fast5.detect_basecall_1d_group(gr)
      
      fast5.basecall_group_descriptions[gr] = desc
    })
  }
}

File.prototype.detect_basecall_event_detection_group = function(gr) {
  var bc_params = this.get_basecall_params(gr)
  var pref = this.event_detection_root_path().substr(1) + "/" + this.event_detection_group_prefix();
  var tmp = bc_params["event_detection"]
  if (tmp) {
    var pref = this.event_detection_root_path().substr(1) + "/" + this.event_detection_group_prefix();
    if (tmp.substr(0, pref.length) === pref) {
      var ed_gr = tmp.substr(pref.length);
      if (this.have_event_detection_group(ed_gr))
        return ed_gr;
    }
  }
  return ""
}

File.prototype.detect_basecall_1d_group = function(gr) {
  var path = this.basecall_group_path(gr) + "/basecall_1d";
  if (this.attribute_exists(path)) {
    var tmp = this.get_attr(path)
    var pref = this.basecall_root_path().substr(1) + "/" + this.basecall_group_prefix();
    if (tmp.length >= pref.length
        && tmp.substr(0, pref.length) == pref)
    {
      var gr_1d = tmp.substr(pref.length);
      if (this.have_basecall_group(gr_1d))
        return gr_1d;
    }
  }
  return gr;
}

// Functions that fill in empty arguments with default values

File.prototype.is_null_arg = function (arg) {
  return typeof(arg) === 'undefined' || arg === ''
}

File.prototype.fill_raw_samples_read_name = function(rn) {
  return (!this.is_null_arg(rn) || !this.raw_samples_read_names.length) ? rn : this.raw_samples_read_names[0]
}

File.prototype.fill_event_detection_group = function(gr) {
  return (!this.is_null_arg(gr) || !this.event_detection_groups.length) ? gr : this.event_detection_groups[0]
}

File.prototype.fill_event_detection_read_name = function(gr, rn) {
  return (!this.is_null_arg(rn) || !this.event_detection_read_names[gr] || !this.event_detection_read_names[gr].length) ? rn : this.event_detection_read_names[gr][0]
}

File.prototype.fill_basecall_group = function(st, gr) {
  return (!this.is_null_arg(gr) || !this.basecall_strand_groups[st]) ? gr : this.basecall_strand_groups[st][0]
}

File.prototype.fill_basecall_1d_group = function(st, gr) {
  var _gr = this.fill_basecall_group(st, gr);
  return this.get_basecall_1d_group(_gr);
}

// Access /file_version
File.prototype.file_version = function() { return this.get_attr (this.file_version_path()) }
// Access /UniqueGlobalKey/channel_id
File.prototype.have_channel_id_params = function() { return this.channel_id_params.sampling_rate > 0 }
File.prototype.get_channel_id_params = function() { return this.channel_id_params }
File.prototype.have_sampling_rate = function() { return this.have_channel_id_params() }
File.prototype.get_sampling_rate = function() { return this.channel_id_params.sampling_rate }
// Access /UniqueGlobalKey/tracking_id
File.prototype.have_tracking_id_params = function() { return this.group_exists(this.tracking_id_path()) }
File.prototype.get_tracking_id_params = function() { return this.file.getDatasetAttributes(this.tracking_id_path()) }
// Access /Sequences
File.prototype.have_sequences_params = function() { return this.group_exists(this.sequences_path()) }
File.prototype.get_sequences_params = function() { return this.file.getDatasetAttributes(this.sequences_path()) }
// Access Raw Samples
File.prototype.get_raw_samples_read_name_list = function() { return this.raw_samples_read_names }
File.prototype.have_raw_samples = function(rn) { return this.is_null_arg(rn) ? this.raw_samples_read_names.length : this.raw_samples_read_names.find (function(s) { return s === rn })  }
File.prototype.get_raw_samples_params = function(rn) { return this.file.getDatasetAttributes(this.raw_samples_params_path (this.fill_raw_samples_read_name(rn))) }
File.prototype.get_raw_int_samples = function(rn) { return this.get_dataset (this.raw_samples_path (this.fill_raw_samples_read_name(rn)))  }
File.prototype.get_raw_samples = function(rn) { return this.get_raw_int_samples(rn).map (this.raw_sample_to_float_function()) }
// Access EventDetection groups
File.prototype.get_event_detection_group_list = function() { return this.event_detection_groups }
File.prototype.have_event_detection_group = function(gr) { return this.is_null_arg(gr) ? Object.keys(this.event_detection_read_names).length : this.event_detection_read_names[gr] }
File.prototype.get_event_detection_read_name_list = function(gr) { return this.event_detection_read_names[this.fill_event_detection_group(gr)] || [] }
File.prototype.get_event_detection_params = function(gr) { return this.file.getDatasetAttributes(this.event_detection_group_path (this.fill_event_detection_group(gr))) }
// Access EventDetection events
File.prototype.have_event_detection_events = function(gr,rn) {
  var _gr = this.fill_event_detection_group(gr)
  var _rn = this.fill_raw_samples_read_name(rn)
  var read_names = this.event_detection_read_names[_gr]
  return read_names && read_names.find (function (s) { return s === _rn })
}
File.prototype.get_event_detection_events_params = function(gr,rn) {
  var _gr = this.fill_event_detection_group(gr)
  var _rn = this.fill_raw_samples_read_name(rn)
  return this.have_event_detection_events(_gr,_rn) ? this.file.getDatasetAttributes(this.event_detection_events_params_path(_gr,_rn)) : undefined
}
File.prototype.get_event_detection_events = function(gr,rn) {
  var _gr = this.fill_event_detection_group(gr)
  var _rn = this.fill_raw_samples_read_name(rn)
  return h5tb.readTable(this.file.id,this.event_detection_events_path(_gr,_rn))
}
// Access Basecall groups
File.prototype.get_basecall_group_list = function() { return this.basecall_groups }
File.prototype.have_basecall_group = function(gr) { return this.is_null_arg(gr) ? this.basecall_groups.length : this.basecall_groups.find (function(s) { return s === gr }) }
File.prototype.get_basecall_strand_group_list = function(st) { return this.basecall_strand_groups[st] }
File.prototype.have_basecall_strand_group = function(st,gr) {
  return this.is_null_arg(gr) ? this.basecall_strand_groups[st].length : this.basecall_strand_groups[st].find (function(s) { return s === gr })
}
File.prototype.get_basecall_group_description = function(gr) { return this.basecall_group_descriptions[gr] }
File.prototype.get_basecall_1d_group = function(gr) { return this.basecall_group_descriptions.hasOwnProperty(gr) ? this.basecall_group_descriptions[gr].bc_1d_gr : undefined }
File.prototype.get_basecall_event_detection_group = function(gr) { return this.basecall_group_descriptions.hasOwnProperty(gr) ? this.basecall_group_descriptions[gr].ed_gr : undefined }
// Access Basecall group params
File.prototype.get_basecall_params = function(gr) { return this.file.getDatasetAttributes (this.basecall_group_path(gr)) }
// Access Basecall group log
File.prototype.have_basecall_log = function(gr) { return this.group_exists (this.basecall_log_path(gr)) }
File.prototype.get_basecall_log = function(gr) { return h5lt.readDataset (this.file.id, this.basecall_log_path(gr)) }
File.prototype.get_basecall_config = function(gr) { return this.group_exists(this.basecall_config_path(gr)) ? this.file.getDatasetAttributes (this.basecall_config_path(gr)) : undefined }
File.prototype.get_basecall_summary = function(gr) { return this.group_exists(this.basecall_summary_path(gr)) ? this.file.getDatasetAttributes (this.basecall_summary_path(gr)) : undefined }
// Access Basecall fastq
File.prototype.have_basecall_fastq = function(st,gr) {
  var _gr = this.fill_event_detection_group(gr)
  return this.basecall_group_descriptions.hasOwnProperty(_gr) && this.basecall_group_descriptions[_gr].have_fastq[st]
}
File.prototype.get_basecall_fastq = function(st,gr) { return h5lt.readDataset (this.file.id, this.basecall_fastq_path(this.fill_event_detection_group(gr),st)) }
File.prototype.have_basecall_seq = function(st,gr) { return this.have_basecall_fastq(st,gr) }
File.prototype.get_basecall_seq = function(st,gr) { return this.fq2seq (this.get_basecall_fastq(st,gr)) }
// Access Basecall events
File.prototype.have_basecall_events = function(st,gr) {
  var _gr = this.fill_basecall_1d_group(st,gr)
  return this.basecall_group_descriptions.hasOwnProperty(_gr) && this.basecall_group_descriptions[_gr].have_events[st]
}
File.prototype.get_basecall_events_params = function(st,gr) { return this.file.getDatasetAttributes (this.basecall_events_path(this.fill_basecall_1d_group(st,gr),st)) }
File.prototype.get_basecall_events = function(st,gr) { return h5tb.readTable (this.file.id, this.basecall_events_path(this.fill_basecall_1d_group(st,gr),st)) }
// Static helpers
File.prototype.time_to_int = function(tf) { return tf * this.channel_id_params.sampling_rate }
File.prototype.time_to_float = function(ti) { return (ti + .5) / this.channel_id_params.sampling_rate }
File.prototype.raw_sample_to_float_function = function() {
  var offset = this.channel_id_params.offset
  var range = this.channel_id_params.range
  var digitisation = this.channel_id_params.digitisation
  var scale = range / digitisation
  return function (si) { return (si + offset) * scale }
}
File.prototype.raw_sample_to_float = function(si) { return this.raw_sample_to_float_function() (si) }
File.prototype.fq2seq = function(fq) { return this.split_fq(fq)[1] }
File.prototype.split_fq = function(fq) { return fq.split('\n') }

exports.File = File
