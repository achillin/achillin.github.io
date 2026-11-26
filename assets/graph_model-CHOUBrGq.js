import{g as e,c as t,a as n}from"./index-BoIQvPqt.js";function r(e,t){for(var n=0;n<t.length;n++){const r=t[n];if("string"!=typeof r&&!Array.isArray(r))for(const t in r)if("default"!==t&&!(t in e)){const n=Object.getOwnPropertyDescriptor(r,t);n&&Object.defineProperty(e,t,n.get?n:{enumerable:!0,get:()=>r[t]})}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class a{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class s{refCount(e){return o("refCount")}incRef(e){return o("incRef")}timerAvailable(){return!0}time(e){return o("time")}read(e){return o("read")}readSync(e){return o("readSync")}readToGPU(e,t){return o("readToGPU")}numDataIds(){return o("numDataIds")}disposeData(e,t){return o("disposeData")}write(e,t,n){return o("write")}move(e,t,n,r,a){return o("move")}createTensorFromGPUData(e,t,n){return o("createTensorFromGPUData")}memory(){return o("memory")}floatPrecision(){return o("floatPrecision")}epsilon(){return 32===this.floatPrecision()?1e-7:1e-4}dispose(){return o("dispose")}}function o(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function i(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,c(e,t,n)}function u(e,t,n){return Math.max(e,Math.min(t,n))}function l(e){return e%2==0?e:e+1}function c(e,t,n){const r=e[t];e[t]=e[n],e[n]=r}function p(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function d(e,t){if(!e)throw new Error("string"==typeof t?t:t())}function h(e,t,n=""){d(y(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function m(e){d(null!=e,()=>"The input to the tensor constructor must be a non-null value.")}function f(e){if(0===e.length)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function g(e,t){if(e===t)return!0;if(null==e||null==t)return!1;if(e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(null!==e[n]&&null!==t[n]&&e[n]!==t[n])return!1;return!0}function y(e,t){if(e===t)return!0;if(null==e||null==t)return!1;if(e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function b(e){return e%1==0}function w(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function N(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function x(e,t=e=>0,n,r){return new Promise((a,s)=>{let o=0;const i=()=>{if(e())return void a();o++;const u=t(o);null!=n&&o>=n?s():null!=r?r(i,u):setTimeout(i,u)};i()})}function v(e,t){let n=1,r=-1;for(let s=0;s<e.length;++s)if(e[s]>=0)n*=e[s];else if(-1===e[s]){if(-1!==r)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${s}`);r=s}else if(e[s]<0)throw Error(`Shapes can not be < 0. Found ${e[s]} at dim ${s}`);if(-1===r){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(0===n)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const a=e.slice();return a[r]=t/n,a}function T(e,t){const n=t.length;return d((e=null==e?t.map((e,t)=>t):[].concat(e)).every(e=>e>=-n&&e<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),d(e.every(e=>b(e)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(e=>e<0?n+e:e)}function S(e,t){const n=[],r=[],a=null!=t&&Array.isArray(t)&&0===t.length,s=null==t||a?null:T(t,e).sort();let o=0;for(let i=0;i<e.length;++i){if(null!=s){if(s[o]===i&&1!==e[i])throw new Error(`Can't squeeze axis ${i} since its dim '${e[i]}' is not 1`);(null==s[o]||s[o]>i)&&1===e[i]&&(n.push(e[i]),r.push(i)),s[o]<=i&&o++}1!==e[i]&&(n.push(e[i]),r.push(i))}return{newShape:n,keptDims:r}}function k(e,t){return E(e,t)}function E(e,t){let n=null;if(null==e||"float32"===e)n=new Float32Array(t);else if("int32"===e)n=new Int32Array(t);else if("bool"===e)n=new Uint8Array(t);else{if("string"!==e)throw new Error(`Unknown data type ${e}`);n=new Array(t)}return n}function _(e,t){for(let n=0;n<e.length;n++){const r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function I(e){return"bool"===e||"complex64"===e||"float32"===e||"int32"===e||"string"===e}function A(e,t){return"complex64"!==t&&(("float32"!==t||"complex64"===e)&&(("int32"!==t||"float32"===e||"complex64"===e)&&("bool"!==t||"bool"!==e)))}function M(e){if("float32"===e||"int32"===e)return 4;if("complex64"===e)return 8;if("bool"===e)return 1;throw new Error(`Unknown dtype ${e}`)}function $(e){if(null==e)return 0;let t=0;return e.forEach(e=>t+=e.length),t}function O(e){return"string"==typeof e||e instanceof String}function D(e){return"boolean"==typeof e}function C(e){return"number"==typeof e}function R(e){return Array.isArray(e)?R(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":C(e)?"float32":O(e)?"string":D(e)?"bool":"float32"}function F(e){return!!(e&&e.constructor&&e.call&&e.apply)}function B(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function z(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function L(e,t,n,r=!1){const a=new Array;if(1===t.length){const s=t[0]*(r?2:1);for(let t=0;t<s;t++)a[t]=n[e+t]}else{const s=t[0],o=t.slice(1),i=o.reduce((e,t)=>e*t)*(r?2:1);for(let t=0;t<s;t++)a[t]=L(e+t*i,o,n,r)}return a}function P(e,t,n=!1){if(0===e.length)return t[0];const r=e.reduce((e,t)=>e*t)*(n?2:1);if(0===r)return[];if(r!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return L(0,e,t,n)}function V(e,t){if(Array.isArray(e))return e;if("float32"===t)return e instanceof Float32Array?e:new Float32Array(e);if("int32"===t)return e instanceof Int32Array?e:new Int32Array(e);if("bool"===t||"string"===t)return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function U(e,t){const n=j(e,t);for(let r=0;r<n.length;r++)n[r]=1;return n}function j(e,t){if(null==t||"float32"===t||"complex64"===t)return new Float32Array(e);if("int32"===t)return new Int32Array(e);if("bool"===t)return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function W(e,t){const n=e.reduce((e,t)=>e*t,1);if(null==t||"float32"===t)return P(e,new Float32Array(n));if("int32"===t)return P(e,new Int32Array(n));if("bool"===t)return P(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)}function q(e){e.forEach(t=>{d(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function G(e,t,n){if(0===t)return 0;if(1===t)return e[0];let r=e[e.length-1];for(let a=0;a<e.length-1;++a)r+=n[a]*e[a];return r}function K(e,t,n){if(0===t)return[];if(1===t)return[e];const r=new Array(t);for(let a=0;a<r.length-1;++a)r[a]=Math.floor(e/n[a]),e-=r[a]*n[a];return r[r.length-1]=e,r}function H(e){return e&&e.then&&"function"==typeof e.then}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const J="tfjsflags";class Z{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=X,this.populateURLFlags()}setPlatform(e,t){null!=this.platform&&!Y().getBool("IS_TEST")&&Y().getBool("PROD"),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},null!=this.urlFlags[e]){const t=this.urlFlags[e];!Y().getBool("IS_TEST")&&Y().getBool("PROD"),this.set(e,t)}}async getAsync(e){return e in this.flags||(this.flags[e]=await this.evaluateFlag(e)),this.flags[e]}get(e){if(e in this.flags)return this.flags[e];const t=this.evaluateFlag(e);if(H(t))throw new Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(null==this.flagRegistry[e])throw new Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,null!=this.flagRegistry[e].setHook&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(null==this.flagRegistry[e])throw new Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(void 0===this.global||void 0===this.global.location||void 0===this.global.location.search)return;const e=this.getQueryParams(this.global.location.search);if(J in e){e[J].split(",").forEach(e=>{const[t,n]=e.split(":");this.urlFlags[t]=function(e,t){const n=t.toLowerCase();return"true"===n||"false"===n?"true"===n:""+ +n===n?+n:t}(0,n)})}}}function X(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...n)=>(function(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}(t,n[0],n[1]),n.join("="))),t}function Y(){return ee}let Q,ee=null;function te(){if(null==Q){let e;if("undefined"!=typeof window)e=window;else if("undefined"!=typeof global)e=global;else if("undefined"!=typeof process)e=process;else{if("undefined"==typeof self)throw new Error("Could not find a global object");e=self}Q=e}return Q}function ne(e,t){const n=function(){const e=te();return null==e._tfGlobals&&(e._tfGlobals=new Map),e._tfGlobals}();if(n.has(e))return n.get(e);{const r=t();return n.set(e,r),n.get(e)}}const re="Abs",ae="Acos",se="Acosh",oe="Add",ie="AddN",ue="All",le="Any",ce="ArgMax",pe="ArgMin",de="Asin",he="Asinh",me="Atan",fe="Atanh",ge="Atan2",ye="AvgPool",be="AvgPoolGrad",we="AvgPool3D",Ne="AvgPool3DGrad",xe="BatchMatMul",ve="BatchToSpaceND",Te="Bincount",Se="BitwiseAnd",ke="BroadcastTo",Ee="BroadcastArgs",_e="Cast",Ie="Ceil",Ae="ClipByValue",Me="Complex",$e="ComplexAbs",Oe="Concat",De="Conv2D",Ce="Conv2DBackpropFilter",Re="Conv2DBackpropInput",Fe="Conv3D",Be="Conv3DBackpropFilterV2",ze="Conv3DBackpropInputV2",Le="Cos",Pe="Cosh",Ve="Cumprod",Ue="Cumsum",je="CropAndResize",We="DenseBincount",qe="DepthToSpace",Ge="DepthwiseConv2dNative",Ke="DepthwiseConv2dNativeBackpropFilter",He="DepthwiseConv2dNativeBackpropInput",Je="Diag",Ze="Dilation2D",Xe="Dilation2DBackpropInput",Ye="Dilation2DBackpropFilter",Qe="Draw",et="RealDiv",tt="Einsum",nt="Elu",rt="EluGrad",at="Erf",st="Equal",ot="Exp",it="ExpandDims",ut="Expm1",lt="FFT",ct="Fill",pt="FlipLeftRight",dt="Floor",ht="FloorDiv",mt="FusedBatchNorm",ft="GatherV2",gt="GatherNd",yt="Greater",bt="GreaterEqual",wt="Identity",Nt="IFFT",xt="Imag",vt="IsFinite",Tt="IsInf",St="IsNan",kt="LeakyRelu",Et="Less",_t="LessEqual",It="LinSpace",At="Log",Mt="Log1p",$t="LogicalAnd",Ot="LogicalNot",Dt="LogicalOr",Ct="LogicalXor",Rt="LogSoftmax",Ft="LowerBound",Bt="LRN",zt="LRNGrad",Lt="MatrixBandPart",Pt="Max",Vt="Maximum",Ut="MaxPool",jt="MaxPoolGrad",Wt="MaxPool3D",qt="MaxPool3DGrad",Gt="MaxPoolWithArgmax",Kt="Mean",Ht="Min",Jt="Minimum",Zt="MirrorPad",Xt="Mod",Yt="Multinomial",Qt="Multiply",en="Neg",tn="NotEqual",nn="NonMaxSuppressionV3",rn="NonMaxSuppressionV4",an="NonMaxSuppressionV5",sn="OnesLike",on="OneHot",un="Pack",ln="PadV2",cn="Pool",pn="Pow",dn="Prelu",hn="Prod",mn="RaggedGather",fn="RaggedRange",gn="RaggedTensorToTensor",yn="Range",bn="Real",wn="Reciprocal",Nn="Relu",xn="Reshape",vn="ResizeNearestNeighbor",Tn="ResizeNearestNeighborGrad",Sn="ResizeBilinear",kn="ResizeBilinearGrad",En="Relu6",_n="Reverse",In="Round",An="Rsqrt",Mn="ScatterNd",$n="TensorScatterUpdate",On="SearchSorted",Dn="Select",Cn="Selu",Rn="Slice",Fn="Sin",Bn="Sinh",zn="Sign",Ln="Sigmoid",Pn="Softplus",Vn="Sqrt",Un="Sum",jn="SpaceToBatchND",Wn="SplitV",qn="Softmax",Gn="SparseFillEmptyRows",Kn="SparseReshape",Hn="SparseSegmentMean",Jn="SparseSegmentSum",Zn="SparseToDense",Xn="SquaredDifference",Yn="Square",Qn="StaticRegexReplace",er="StridedSlice",tr="StringNGrams",nr="StringSplit",rr="StringToHashBucketFast",ar="Sub",sr="Tan",or="Tanh",ir="Tile",ur="TopK",lr="Transform",cr="Transpose",pr="Unique",dr="Unpack",hr="UnsortedSegmentSum",mr="UpperBound",fr="ZerosLike",gr="Step",yr="FromPixels",br="RotateWithOffset",wr="_FusedMatMul",Nr="FusedConv2D",xr="FusedDepthwiseConv2D";
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function vr(...e){!Y().getBool("IS_TEST")&&Y().getBool("PROD")}function Tr(...e){!Y().getBool("IS_TEST")&&Y().getBool("PROD")}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sr=ne("kernelRegistry",()=>new Map),kr=ne("gradRegistry",()=>new Map);function Er(e,t){const n=Cr(e,t);return Sr.get(n)}function _r(e){return kr.get(e)}function Ir(e){const t=Sr.entries(),n=[];for(;;){const{done:r,value:a}=t.next();if(r)break;const[s,o]=a,[i]=s.split("_");i===e&&n.push(o)}return n}function Ar(e){const{kernelName:t,backendName:n}=e,r=Cr(t,n);Sr.has(r)&&vr(),Sr.set(r,e)}function Mr(e){const{kernelName:t}=e;kr.has(t)&&Y().getBool("DEBUG")&&vr(),kr.set(t,e)}function $r(e,t){const n=Cr(e,t);if(!Sr.has(n))throw new Error(`The kernel '${e}' for backend '${t}' is not registered`);Sr.delete(n)}function Or(e){if(!kr.has(e))throw new Error(`The gradient '${e}' for backend is not registered`);kr.delete(e)}function Dr(e,t){Ir(e).forEach(e=>{Ar(Object.assign({},e,{backendName:t}))})}function Cr(e,t){return`${t}_${e}`}
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rr(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var Fr=zr,Br=null;try{Br=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch(vf){}function zr(e,t,n){this.low=0|e,this.high=0|t,this.unsigned=!!n}function Lr(e){return!0===(e&&e.__isLong__)}zr.prototype.__isLong__,Object.defineProperty(zr.prototype,"__isLong__",{value:!0}),zr.isLong=Lr;var Pr={},Vr={};function Ur(e,t){var n,r,a;return t?(a=0<=(e>>>=0)&&e<256)&&(r=Vr[e])?r:(n=Wr(e,(0|e)<0?-1:0,!0),a&&(Vr[e]=n),n):(a=-128<=(e|=0)&&e<128)&&(r=Pr[e])?r:(n=Wr(e,e<0?-1:0,!1),a&&(Pr[e]=n),n)}function jr(e,t){if(isNaN(e))return t?Qr:Yr;if(t){if(e<0)return Qr;if(e>=Jr)return aa}else{if(e<=-Zr)return sa;if(e+1>=Zr)return ra}return e<0?jr(-e,t).neg():Wr(e%Hr|0,e/Hr|0,t)}function Wr(e,t,n){return new zr(e,t,n)}zr.fromInt=Ur,zr.fromNumber=jr,zr.fromBits=Wr;var qr=Math.pow;function Gr(e,t,n){if(0===e.length)throw Error("empty string");if("NaN"===e||"Infinity"===e||"+Infinity"===e||"-Infinity"===e)return Yr;if("number"==typeof t?(n=t,t=!1):t=!!t,(n=n||10)<2||36<n)throw RangeError("radix");var r;if((r=e.indexOf("-"))>0)throw Error("interior hyphen");if(0===r)return Gr(e.substring(1),t,n).neg();for(var a=jr(qr(n,8)),s=Yr,o=0;o<e.length;o+=8){var i=Math.min(8,e.length-o),u=parseInt(e.substring(o,o+i),n);if(i<8){var l=jr(qr(n,i));s=s.mul(l).add(jr(u))}else s=(s=s.mul(a)).add(jr(u))}return s.unsigned=t,s}function Kr(e,t){return"number"==typeof e?jr(e,t):"string"==typeof e?Gr(e,t):Wr(e.low,e.high,"boolean"==typeof t?t:e.unsigned)}zr.fromString=Gr,zr.fromValue=Kr;var Hr=4294967296,Jr=Hr*Hr,Zr=Jr/2,Xr=Ur(1<<24),Yr=Ur(0);zr.ZERO=Yr;var Qr=Ur(0,!0);zr.UZERO=Qr;var ea=Ur(1);zr.ONE=ea;var ta=Ur(1,!0);zr.UONE=ta;var na=Ur(-1);zr.NEG_ONE=na;var ra=Wr(-1,2147483647,!1);zr.MAX_VALUE=ra;var aa=Wr(-1,-1,!0);zr.MAX_UNSIGNED_VALUE=aa;var sa=Wr(0,-2147483648,!1);zr.MIN_VALUE=sa;var oa=zr.prototype;oa.toInt=function(){return this.unsigned?this.low>>>0:this.low},oa.toNumber=function(){return this.unsigned?(this.high>>>0)*Hr+(this.low>>>0):this.high*Hr+(this.low>>>0)},oa.toString=function(e){if((e=e||10)<2||36<e)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative()){if(this.eq(sa)){var t=jr(e),n=this.div(t),r=n.mul(t).sub(this);return n.toString(e)+r.toInt().toString(e)}return"-"+this.neg().toString(e)}for(var a=jr(qr(e,6),this.unsigned),s=this,o="";;){var i=s.div(a),u=(s.sub(i.mul(a)).toInt()>>>0).toString(e);if((s=i).isZero())return u+o;for(;u.length<6;)u="0"+u;o=""+u+o}},oa.getHighBits=function(){return this.high},oa.getHighBitsUnsigned=function(){return this.high>>>0},oa.getLowBits=function(){return this.low},oa.getLowBitsUnsigned=function(){return this.low>>>0},oa.getNumBitsAbs=function(){if(this.isNegative())return this.eq(sa)?64:this.neg().getNumBitsAbs();for(var e=0!=this.high?this.high:this.low,t=31;t>0&&!(e&1<<t);t--);return 0!=this.high?t+33:t+1},oa.isZero=function(){return 0===this.high&&0===this.low},oa.eqz=oa.isZero,oa.isNegative=function(){return!this.unsigned&&this.high<0},oa.isPositive=function(){return this.unsigned||this.high>=0},oa.isOdd=function(){return!(1&~this.low)},oa.isEven=function(){return!(1&this.low)},oa.equals=function(e){return Lr(e)||(e=Kr(e)),(this.unsigned===e.unsigned||this.high>>>31!=1||e.high>>>31!=1)&&(this.high===e.high&&this.low===e.low)},oa.eq=oa.equals,oa.notEquals=function(e){return!this.eq(e)},oa.neq=oa.notEquals,oa.ne=oa.notEquals,oa.lessThan=function(e){return this.comp(e)<0},oa.lt=oa.lessThan,oa.lessThanOrEqual=function(e){return this.comp(e)<=0},oa.lte=oa.lessThanOrEqual,oa.le=oa.lessThanOrEqual,oa.greaterThan=function(e){return this.comp(e)>0},oa.gt=oa.greaterThan,oa.greaterThanOrEqual=function(e){return this.comp(e)>=0},oa.gte=oa.greaterThanOrEqual,oa.ge=oa.greaterThanOrEqual,oa.compare=function(e){if(Lr(e)||(e=Kr(e)),this.eq(e))return 0;var t=this.isNegative(),n=e.isNegative();return t&&!n?-1:!t&&n?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1},oa.comp=oa.compare,oa.negate=function(){return!this.unsigned&&this.eq(sa)?sa:this.not().add(ea)},oa.neg=oa.negate,oa.add=function(e){Lr(e)||(e=Kr(e));var t=this.high>>>16,n=65535&this.high,r=this.low>>>16,a=65535&this.low,s=e.high>>>16,o=65535&e.high,i=e.low>>>16,u=0,l=0,c=0,p=0;return c+=(p+=a+(65535&e.low))>>>16,l+=(c+=r+i)>>>16,u+=(l+=n+o)>>>16,u+=t+s,Wr((c&=65535)<<16|(p&=65535),(u&=65535)<<16|(l&=65535),this.unsigned)},oa.subtract=function(e){return Lr(e)||(e=Kr(e)),this.add(e.neg())},oa.sub=oa.subtract,oa.multiply=function(e){if(this.isZero())return Yr;if(Lr(e)||(e=Kr(e)),Br)return Wr(Br.mul(this.low,this.high,e.low,e.high),Br.get_high(),this.unsigned);if(e.isZero())return Yr;if(this.eq(sa))return e.isOdd()?sa:Yr;if(e.eq(sa))return this.isOdd()?sa:Yr;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(Xr)&&e.lt(Xr))return jr(this.toNumber()*e.toNumber(),this.unsigned);var t=this.high>>>16,n=65535&this.high,r=this.low>>>16,a=65535&this.low,s=e.high>>>16,o=65535&e.high,i=e.low>>>16,u=65535&e.low,l=0,c=0,p=0,d=0;return p+=(d+=a*u)>>>16,c+=(p+=r*u)>>>16,p&=65535,c+=(p+=a*i)>>>16,l+=(c+=n*u)>>>16,c&=65535,l+=(c+=r*i)>>>16,c&=65535,l+=(c+=a*o)>>>16,l+=t*u+n*i+r*o+a*s,Wr((p&=65535)<<16|(d&=65535),(l&=65535)<<16|(c&=65535),this.unsigned)},oa.mul=oa.multiply,oa.divide=function(e){if(Lr(e)||(e=Kr(e)),e.isZero())throw Error("division by zero");var t,n,r;if(Br)return this.unsigned||-2147483648!==this.high||-1!==e.low||-1!==e.high?Wr((this.unsigned?Br.div_u:Br.div_s)(this.low,this.high,e.low,e.high),Br.get_high(),this.unsigned):this;if(this.isZero())return this.unsigned?Qr:Yr;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return Qr;if(e.gt(this.shru(1)))return ta;r=Qr}else{if(this.eq(sa))return e.eq(ea)||e.eq(na)?sa:e.eq(sa)?ea:(t=this.shr(1).div(e).shl(1)).eq(Yr)?e.isNegative()?ea:na:(n=this.sub(e.mul(t)),r=t.add(n.div(e)));if(e.eq(sa))return this.unsigned?Qr:Yr;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();r=Yr}for(n=this;n.gte(e);){t=Math.max(1,Math.floor(n.toNumber()/e.toNumber()));for(var a=Math.ceil(Math.log(t)/Math.LN2),s=a<=48?1:qr(2,a-48),o=jr(t),i=o.mul(e);i.isNegative()||i.gt(n);)i=(o=jr(t-=s,this.unsigned)).mul(e);o.isZero()&&(o=ea),r=r.add(o),n=n.sub(i)}return r},oa.div=oa.divide,oa.modulo=function(e){return Lr(e)||(e=Kr(e)),Br?Wr((this.unsigned?Br.rem_u:Br.rem_s)(this.low,this.high,e.low,e.high),Br.get_high(),this.unsigned):this.sub(this.div(e).mul(e))},oa.mod=oa.modulo,oa.rem=oa.modulo,oa.not=function(){return Wr(~this.low,~this.high,this.unsigned)},oa.and=function(e){return Lr(e)||(e=Kr(e)),Wr(this.low&e.low,this.high&e.high,this.unsigned)},oa.or=function(e){return Lr(e)||(e=Kr(e)),Wr(this.low|e.low,this.high|e.high,this.unsigned)},oa.xor=function(e){return Lr(e)||(e=Kr(e)),Wr(this.low^e.low,this.high^e.high,this.unsigned)},oa.shiftLeft=function(e){return Lr(e)&&(e=e.toInt()),0==(e&=63)?this:e<32?Wr(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):Wr(0,this.low<<e-32,this.unsigned)},oa.shl=oa.shiftLeft,oa.shiftRight=function(e){return Lr(e)&&(e=e.toInt()),0==(e&=63)?this:e<32?Wr(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):Wr(this.high>>e-32,this.high>=0?0:-1,this.unsigned)},oa.shr=oa.shiftRight,oa.shiftRightUnsigned=function(e){if(Lr(e)&&(e=e.toInt()),0===(e&=63))return this;var t=this.high;return e<32?Wr(this.low>>>e|t<<32-e,t>>>e,this.unsigned):Wr(32===e?t:t>>>e-32,0,this.unsigned)},oa.shru=oa.shiftRightUnsigned,oa.shr_u=oa.shiftRightUnsigned,oa.toSigned=function(){return this.unsigned?Wr(this.low,this.high,!1):this},oa.toUnsigned=function(){return this.unsigned?this:Wr(this.low,this.high,!0)},oa.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()},oa.toBytesLE=function(){var e=this.high,t=this.low;return[255&t,t>>>8&255,t>>>16&255,t>>>24,255&e,e>>>8&255,e>>>16&255,e>>>24]},oa.toBytesBE=function(){var e=this.high,t=this.low;return[e>>>24,e>>>16&255,e>>>8&255,255&e,t>>>24,t>>>16&255,t>>>8&255,255&t]},zr.fromBytes=function(e,t,n){return n?zr.fromBytesLE(e,t):zr.fromBytesBE(e,t)},zr.fromBytesLE=function(e,t){return new zr(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,t)},zr.fromBytesBE=function(e,t){return new zr(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],t)};const ia=e(Fr),ua=ia||r({__proto__:null,default:ia},[Fr]);function la(e){return ua.fromString(e,!0,16)}const ca=la("c3a5c85c97cb3127"),pa=la("b492b66fbe98f273"),da=la("9ae16a3b2f90404f");function ha(e){return e.xor(e.shru(47))}function ma(e,t,n){const r=e.slice(t,t+n);return ua.fromBytes(Array.from(r),!0,!0)}function fa(e,t){return ma(e,t,8)}function ga(e,t){return ma(e,t,4)}function ya(e,t){return 0===t?e:e.shru(t).or(e.shl(64-t))}function ba(e,t,n=la("9ddfea08eb382d69")){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let a=t.xor(r).mul(n);return a=a.xor(a.shru(47)),a=a.mul(n),a}function wa(e,t,n,r){return function(e,t,n,r,a,s){a=a.add(e),s=ya(s.add(a).add(r),21);const o=a;return a=(a=a.add(t)).add(n),s=s.add(ya(a,44)),[a.add(r),s.add(o)]}(fa(e,t),fa(e,t+8),fa(e,t+16),fa(e,t+24),n,r)}function Na(e,t=e.length){const n=ua.fromNumber(81,!0);if(t<=32)return t<=16?function(e,t=e.length){if(t>=8){const n=da.add(2*t),r=fa(e,0).add(da),a=fa(e,t-8);return ba(ya(a,37).mul(n).add(r),ya(r,25).add(a).mul(n),n)}if(t>=4){const n=da.add(2*t);return ba(ga(e,0).shl(3).add(t),ga(e,t-4),n)}if(t>0){const n=e[0]+(e[t>>1]<<8),r=t+(e[t-1]<<2);return ha(da.mul(n).xor(ca.mul(r))).mul(da)}return da}(e,t):function(e,t=e.length){const n=da.add(2*t),r=fa(e,0).mul(pa),a=fa(e,8),s=fa(e,t-8).mul(n),o=fa(e,t-16).mul(da);return ba(ya(r.add(a),43).add(ya(s,30)).add(o),r.add(ya(a.add(da),18)).add(s),n)}(e,t);if(t<=64)return function(e,t=e.length){const n=da.add(2*t),r=fa(e,0).mul(da),a=fa(e,8),s=fa(e,t-8).mul(n),o=fa(e,t-16).mul(da),i=ya(r.add(a),43).add(ya(s,30)).add(o),u=ba(i,r.add(ya(a.add(da),18)).add(s),n),l=fa(e,16).mul(n),c=fa(e,24),p=i.add(fa(e,t-32)).mul(n),d=u.add(fa(e,t-24)).mul(n);return ba(ya(l.add(c),43).add(ya(p,30)).add(d),l.add(ya(c.add(r),18)).add(p),n)}(e,t);let r=n,a=n.mul(pa).add(113),s=ha(a.mul(da).add(113)).mul(da),o=[ua.UZERO,ua.UZERO],i=[ua.UZERO,ua.UZERO];r=r.mul(da).add(fa(e,0));let u=0;const l=64*(t-1>>6),c=l+(t-1&63)-63;do{r=ya(r.add(a).add(o[0]).add(fa(e,u+8)),37).mul(pa),a=ya(a.add(o[1]).add(fa(e,u+48)),42).mul(pa),r=r.xor(i[1]),a=a.add(o[0]).add(fa(e,u+40)),s=ya(s.add(i[0]),33).mul(pa),o=wa(e,u,o[1].mul(pa),r.add(i[0])),i=wa(e,u+32,s.add(i[1]),a.add(fa(e,u+16))),[s,r]=[r,s],u+=64}while(u!==l);const p=pa.add(s.and(255).shl(1));return u=c,i[0]=i[0].add(t-1&63),o[0]=o[0].add(i[0]),i[0]=i[0].add(o[0]),r=ya(r.add(a).add(o[0]).add(fa(e,u+8)),37).mul(p),a=ya(a.add(o[1]).add(fa(e,u+48)),42).mul(p),r=r.xor(i[1].mul(9)),a=a.add(o[0].mul(9).add(fa(e,u+40))),s=ya(s.add(i[0]),33).mul(p),o=wa(e,u,o[1].mul(p),r.add(i[0])),i=wa(e,u+32,s.add(i[1]),a.add(fa(e,u+16))),[s,r]=[r,s],ba(ba(o[0],i[0],p).add(ha(a).mul(ca)).add(s),ba(o[1],i[1],p).add(r),p)}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xa(e,t){return"string"===t?ka(e):va([e],t)}function va(e,t){if("string"===t)throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=Ia(e)),Y().getBool("DEBUG")&&_(e,t),function(e,t){return e instanceof Float32Array&&"float32"===t||e instanceof Int32Array&&"int32"===t||e instanceof Uint8Array&&"bool"===t}(e,t))return e;if(null==t||"float32"===t||"complex64"===t)return new Float32Array(e);if("int32"===t)return new Int32Array(e);if("bool"===t){const t=new Uint8Array(e.length);for(let n=0;n<t.length;++n)0!==Math.round(e[n])&&(t[n]=1);return t}throw new Error(`Unknown data type ${t}`)}function Ta(){return Y().platform.now()}function Sa(e,t){return Y().platform.fetch(e,t)}function ka(e,t="utf-8"){return t=t||"utf-8",Y().platform.encode(e,t)}function Ea(e,t="utf-8"){return t=t||"utf-8",Y().platform.decode(e,t)}function _a(e){return null!=Y().platform.isTypedArray?Y().platform.isTypedArray(e):Rr(e)}function Ia(e,t=[],n=!1){if(null==t&&(t=[]),"boolean"==typeof e||"number"==typeof e||"string"==typeof e||H(e)||null==e||_a(e)&&n)t.push(e);else if(Array.isArray(e)||_a(e))for(let r=0;r<e.length;++r)Ia(e[r],t,n);else{let r=-1;for(const t of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(t)&&(r=Math.max(r,Number(t)));for(let a=0;a<=r;a++)Ia(e[a],t,n)}return t}const Aa=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:y,arraysEqualWithNull:g,assert:d,assertNonNegativeIntegerDimensions:q,assertNonNull:m,assertShapesMatch:h,bytesFromStringArray:$,bytesPerElement:M,checkConversionForErrors:_,clamp:u,computeStrides:z,convertBackendValuesAndArrayBuffer:V,createScalarValue:xa,createShuffledIndices:function(e){const t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return i(t),t},decodeString:Ea,distSquared:function(e,t){let n=0;for(let r=0;r<e.length;r++){const a=Number(e[r])-Number(t[r]);n+=a*a}return n},encodeString:ka,fetch:Sa,fingerPrint64:Na,flatten:Ia,getArrayFromDType:E,getTypedArrayFromDType:k,hasEncodingLoss:A,hexToLong:la,indexToLoc:K,inferDtype:R,inferFromImplicitShape:v,isBoolean:D,isFunction:F,isInt:b,isNumber:C,isPromise:H,isScalarShape:function(e){return 0===e.length},isString:O,isTypedArray:_a,isValidDtype:I,locToIndex:G,makeOnesTypedArray:U,makeZerosNestedTypedArray:W,makeZerosTypedArray:j,nearestDivisor:B,nearestLargerEven:l,now:Ta,parseAxisParam:T,randUniform:function(e,t){const n=Math.random();return t*n+(1-n)*e},repeatedTry:x,rightPad:N,shuffle:i,shuffleCombo:function(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,r=0;for(;n>0;)r=Math.random()*n|0,n--,c(e,n,r),c(t,n,r)},sizeFromShape:f,sizeToSquarishShape:w,squeezeShape:S,sum:p,swap:c,tanh:function(e){if(null!=Math.tanh)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{const t=Math.exp(2*e);return(t-1)/(t+1)}},toNestedArray:P,toTypedArray:va},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ma{constructor(e,t){this.backendTimer=e,this.logger=t,null==t&&(this.logger=new Oa)}profileKernel(e,t,n){let r;const a=()=>{r=n()};let s;const o=Ta();if(this.backendTimer.timerAvailable())s=this.backendTimer.time(a);else{a();for(const e of r)e.dataSync();s=Promise.resolve({kernelMs:Ta()-o})}if(Y().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let i=0;i<r.length;i++){const t=r[i];t.data().then(n=>{$a(n,t.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:s.then(e=>e.kernelMs),extraInfo:s.then(e=>null!=e.getExtraProfileInfo?e.getExtraProfileInfo():"")}}logKernelProfile(e){const{kernelName:t,outputs:n,timeMs:r,inputs:a,extraInfo:s}=e;n.forEach(e=>{Promise.all([e.data(),r,s]).then(n=>{this.logger.logKernelProfile(t,e,n[0],n[1],a,n[2])})})}}function $a(e,t,n){if("float32"!==t)return!1;for(let r=0;r<e.length;r++){const t=e[r];if(isNaN(t)||!isFinite(t))return!0}return!1}class Oa{logKernelProfile(e,t,n,r,a,s){"number"==typeof r?N(`${r}ms`,9):r.error,N(e,25),t.rank,t.size,N(t.shape.toString(),14);let o="";for(const i in a){const e=a[i];if(null!=e){const n=e.shape||t.shape,r=n.length;o+=`${i}: ${r}D ${r>0?n:""} `}}}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Da(e,t,n,r){const a=z(t),s=function(e,t,n,r){const a=f(t),s=r[r.length-1],o=new Array(s).fill(0),i=t.length,u="complex64"===n?Ba(e):e;if(i>1)for(let l=0;l<a/s;l++){const e=l*s;for(let t=0;t<s;t++)o[t]=Math.max(o[t],Ca(u[e+t],0,n).length)}return o}(e,t,n,a),o=t.length,i=Fa(e,t,n,a,s),u=["Tensor"];return r&&(u.push(`  dtype: ${n}`),u.push(`  rank: ${o}`),u.push(`  shape: [${t}]`),u.push("  values:")),u.push(i.map(e=>"    "+e).join("\n")),u.join("\n")}function Ca(e,t,n){let r;return r=Array.isArray(e)?`${parseFloat(e[0].toFixed(7))} + ${parseFloat(e[1].toFixed(7))}j`:O(e)?`'${e}'`:"bool"===n?Ra(e):parseFloat(e.toFixed(7)).toString(),N(r,t)}function Ra(e){return 0===e?"false":"true"}function Fa(e,t,n,r,a,s=!0){const o="complex64"===n?2:1,i=t[0],u=t.length;if(0===u){if("complex64"===n){return[Ca(Ba(e)[0],0,n)]}return"bool"===n?[Ra(e[0])]:[e[0].toString()]}if(1===u){if(i>20){const t=3*o;let r=Array.from(e.slice(0,t)),s=Array.from(e.slice((i-3)*o,i*o));return"complex64"===n&&(r=Ba(r),s=Ba(s)),["["+r.map((e,t)=>Ca(e,a[t],n)).join(", ")+", ..., "+s.map((e,t)=>Ca(e,a[i-3+t],n)).join(", ")+"]"]}return["["+("complex64"===n?Ba(e):Array.from(e)).map((e,t)=>Ca(e,a[t],n)).join(", ")+"]"]}const l=t.slice(1),c=r.slice(1),p=r[0]*o,d=[];if(i>20){for(let t=0;t<3;t++){const r=t*p,s=r+p;d.push(...Fa(e.slice(r,s),l,n,c,a,!1))}d.push("...");for(let t=i-3;t<i;t++){const r=t*p,s=r+p;d.push(...Fa(e.slice(r,s),l,n,c,a,t===i-1))}}else for(let f=0;f<i;f++){const t=f*p,r=t+p;d.push(...Fa(e.slice(t,r),l,n,c,a,f===i-1))}const h=2===u?",":"";d[0]="["+(i>0?d[0]+h:"");for(let f=1;f<d.length-1;f++)d[f]=" "+d[f]+h;let m=",\n";for(let f=2;f<u;f++)m+="\n";return d[d.length-1]=" "+d[d.length-1]+"]"+(s?"":m),d}function Ba(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class za{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=f(e),null!=n){const e=n.length;d(e===this.size,()=>`Length of values '${e}' does not match the size inferred by the shape '${this.size}'.`)}if("complex64"===t)throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=n||E(t,this.size),this.strides=z(e)}set(e,...t){0===t.length&&(t=[0]),d(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);const n=this.locToIndex(t);this.values[n]=e}get(...e){0===e.length&&(e=[0]);let t=0;for(const r of e){if(r<0||r>=this.shape[t]){const t=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw new Error(t)}t++}let n=e[e.length-1];for(let r=0;r<e.length-1;++r)n+=this.strides[r]*e[r];return this.values[n]}locToIndex(e){if(0===this.rank)return 0;if(1===this.rank)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(0===this.rank)return[];if(1===this.rank)return[e];const t=new Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return La().makeTensor(this.values,this.shape,this.dtype)}}let La=null,Pa=null;class Va{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||"float32",this.size=f(e),this.strides=z(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const e=await this.data();return Pa.buffer(this.shape,this.dtype,e)}bufferSync(){return Pa.buffer(this.shape,this.dtype,this.dataSync())}async array(){const e=await this.data();return P(this.shape,e,"complex64"===this.dtype)}arraySync(){return P(this.shape,this.dataSync(),"complex64"===this.dtype)}async data(){this.throwIfDisposed();const e=La().read(this.dataId);if("string"===this.dtype){const n=await e;try{return n.map(e=>Ea(e))}catch(t){throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),La().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();const e=La().readSync(this.dataId);if("string"===this.dtype)try{return e.map(e=>Ea(e))}catch(t){throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();const e=await La().read(this.dataId);return"string"===this.dtype?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),La().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(e=!1){return Pa.print(this,e)}clone(){return this.throwIfDisposed(),Pa.clone(this)}toString(e=!1){return Da(this.dataSync(),this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Pa.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),La().makeVariable(this,e,t,n)}}function Ua(){return ne("Tensor",()=>Va)}Object.defineProperty(Va,Symbol.hasInstance,{value:e=>!!e&&null!=e.data&&null!=e.dataSync&&null!=e.throwIfDisposed}),Ua();class ja extends Va{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw new Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!y(e.shape,this.shape))throw new Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);La().disposeTensor(this),this.dataId=e.dataId,La().incRef(this,null)}dispose(){La().disposeVariable(this),this.isDisposedInternal=!0}}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
var Wa,qa,Ga,Ka,Ha,Ja,Za,Xa,Ya,Qa;Object.defineProperty(ja,Symbol.hasInstance,{value:e=>e instanceof Va&&null!=e.assign&&e.assign instanceof Function}),(qa=Wa||(Wa={})).R0="R0",qa.R1="R1",qa.R2="R2",qa.R3="R3",qa.R4="R4",qa.R5="R5",qa.R6="R6",(Ka=Ga||(Ga={})).float32="float32",Ka.int32="int32",Ka.bool="int32",Ka.complex64="complex64",(Ja=Ha||(Ha={})).float32="float32",Ja.int32="int32",Ja.bool="bool",Ja.complex64="complex64",(Xa=Za||(Za={})).float32="float32",Xa.int32="float32",Xa.bool="float32",Xa.complex64="complex64",(Qa=Ya||(Ya={})).float32="complex64",Qa.int32="complex64",Qa.bool="complex64",Qa.complex64="complex64";const es={float32:Za,int32:Ga,bool:Ha,complex64:Ya};function ts(e,t){if("string"===e||"string"===t){if("string"===e&&"string"===t)return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return es[e][t]}function ns(e){return ts(e,"int32")}function rs(e){return null!=e&&"object"==typeof e&&"texture"in e&&e.texture instanceof WebGLTexture}function as(e){return"undefined"!=typeof GPUBuffer&&null!=e&&"object"==typeof e&&"buffer"in e&&e.buffer instanceof GPUBuffer}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ss(e,t){if(e.dtype===t.dtype)return[e,t];const n=ts(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function os(e,t){d(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function is(e,t){return t.some(t=>t.id===e.id)}function us(e){const t=[];return ls(e,t,new Set),t}function ls(e,t,n){if(null==e)return;if(e instanceof Va)return void t.push(e);if(r=e,!Array.isArray(r)&&"object"!=typeof r)return;var r;const a=e;for(const s in a){const e=a[s];n.has(e)||(n.add(e),ls(e,t,n))}}const cs=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:os,getTensorsInContainer:us,isTensorInList:is,makeTypesMatch:ss},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ps(e){return null!=e.kernelName}class ds{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(const e in this.registeredVariables)this.registeredVariables[e].dispose()}}class hs{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new ds}async ready(){if(null!=this.pendingBackendInit)return this.pendingBackendInit.then(()=>{});if(null!=this.backendInstance)return;const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t];if(await this.initializeBackend(n).success)return void(await this.setBackend(n))}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(null!=this.pendingBackendInit)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(null==this.backendInstance){const{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw new Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry)){if(!(e in this.registryFactory))return null;{const{asyncInit:t}=this.initializeBackend(e);if(t)return null}}return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(vr(),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(null==this.registryFactory[e])throw new Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,null==this.registry[e]){this.backendInstance=null;const{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new Ma(this.backendInstance),!0}setupRegisteredKernels(){Ir(this.backendName).forEach(e=>{null!=e.setupFunc&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){Ir(e).forEach(t=>{null!=t.disposeFunc&&t.disposeFunc(this.registry[e])})}initializeBackend(e){const t=this.registryFactory[e];if(null==t)throw new Error(`Cannot initialize backend ${e}, no registration found.`);try{const n=t.factory();if(!n||n instanceof s||"function"!=typeof n.then)return this.registry[e]=n,{success:!0,asyncInit:!1};{const t=++this.pendingBackendInitId,r=n.then(n=>!(t<this.pendingBackendInitId)&&(this.registry[e]=n,this.pendingBackendInit=null,!0)).catch(e=>(t<this.pendingBackendInitId||(this.pendingBackendInit=null,vr(),vr(e.stack||e.message)),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}}catch(n){return vr(),vr(n.stack||n.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw new Error(`${e} backend not found in registry`);this.backendName===e&&null!=this.pendingBackendInit&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(0===Object.keys(this.registryFactory).length)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t],{success:r,asyncInit:a}=this.initializeBackend(n);if(a||r)return{name:n,asyncInit:a}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,t){const n=this.state.tensorInfo.get(t),r=n.backend,a=this.readSync(t),s=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,a,n.shape,n.dtype,s),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n,r=null;if(null==t){if("function"!=typeof e)throw new Error("Please provide a function to tidy()");t=e}else{if("string"!=typeof e&&!(e instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if("function"!=typeof t)throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");r=e}return this.scopedRun(()=>this.startScope(r),()=>this.endScope(n),()=>(n=t(),n))}scopedRun(e,t,n){e();try{const e=n();return t(),e}catch(r){throw t(),r}}nextTensorId(){return hs.nextTensorId++}nextVariableId(){return hs.nextVariableId++}clone(e){const t=fs.runKernel(wt,{x:e}),n={x:e};return this.addTapeNode(this.state.activeScope.name,n,[t],e=>({x:()=>{const t={x:e},n={dtype:"float32"};return fs.runKernel(_e,t,n)}}),[],{}),t}runKernel(e,t,n){null==this.backendName&&this.backend;if(!(null!=Er(e,this.backendName)))throw new Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,t,n){const r=this.backend.numDataIds();let a=0;n.forEach(e=>{a+="complex64"===e.dtype?3:1});const s=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],o=r-t-a-s;if(o>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${o} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[];const r=this.isTapeOn(),a=this.state.numBytes,s=this.state.numTensors;let o,i;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0),null==this.backendName&&this.backend;const u=ps(e)?e.kernelName:null!=this.state.activeScope?this.state.activeScope.name:"";if(ps(e)){const{kernelName:t,inputs:a,attrs:s}=e;null==this.backendName&&this.backend;const u=Er(t,this.backendName);d(null!=u,()=>`Cannot find registered kernel '${t}' for backend '${this.backendName}'`),o=()=>{const e=this.backend.numDataIds();i=u.kernelFunc({inputs:a,attrs:s,backend:this.backend});const o=Array.isArray(i)?i:[i];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(t,e,o);const l=o.map(e=>null!=e.rank?e:this.makeTensorFromTensorInfo(e));if(r){const e=this.getTensorsForGradient(t,a,l);n=this.saveTensorsForBackwardMode(e)}return l}}else{const{forwardFunc:t}=e,a=e=>{r&&(n=e.map(e=>this.keep(this.clone(e))))};o=()=>{const e=this.backend.numDataIds();i=this.tidy(()=>t(this.backend,a));const n=Array.isArray(i)?i:[i];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(u,e,n),n}}const{inputs:l,attrs:c}=e,p=ps(e)?null:e.backwardsFunc;let h;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{this.ENV.getBool("DEBUG")||this.state.profiling?(h=this.profiler.profileKernel(u,l,()=>o()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(h),t=h.outputs):t=o()}),r&&this.addTapeNode(u,l,t,p,n,c),this.state.profiling&&this.state.activeProfile.kernels.push({name:u,bytesAdded:this.state.numBytes-a,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-s,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(l).map(e=>null!=l[e]?l[e].shape:null),outputShapes:t.map(e=>e.shape),kernelTimeMs:h.timeMs,extraInfo:h.extraInfo}),Array.isArray(i)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(e=>this.keep(this.clone(e)))}getTensorsForGradient(e,t,n){const r=_r(e);if(null!=r){const e=r.inputsToSave||[],a=r.outputsToSave||[];let s;r.saveAllInputs?(d(Array.isArray(t),()=>"saveAllInputs is true, expected inputs to be an array."),s=Object.keys(t).map(e=>t[e])):s=e.map(e=>t[e]);const o=n.filter((e,t)=>a[t]);return s.concat(o)}return[]}makeTensor(e,t,n,r){if(null==e)throw new Error("Values passed to engine.makeTensor() are null");n=n||"float32",r=r||this.backend;let a=e;"string"===n&&O(e[0])&&(a=e.map(e=>ka(e)));const s=r.write(a,t,n),o=new Va(t,n,s,this.nextTensorId());if(this.trackTensor(o,r),"string"===n){const e=this.state.tensorInfo.get(s),t=$(a);this.state.numBytes+=t-e.bytes,e.bytes=t}return o}makeTensorFromDataId(e,t,n,r){const a={dataId:e,shape:t,dtype:n=n||"float32"};return this.makeTensorFromTensorInfo(a,r)}makeTensorFromTensorInfo(e,t){const{dataId:n,shape:r,dtype:a}=e,s=new Va(r,a,n,this.nextTensorId());return this.trackTensor(s,t),s}makeVariable(e,t=!0,n,r){n=n||this.nextVariableId().toString(),null!=r&&r!==e.dtype&&(e=e.cast(r));const a=new ja(e,t,n,this.nextTensorId());if(null!=this.state.registeredVariables[a.name])throw new Error(`Variable with name ${a.name} was already registered`);return this.state.registeredVariables[a.name]=a,this.incRef(a,this.backend),a}trackTensor(e,t){this.state.numTensors++,"string"===e.dtype&&this.state.numStringTensors++;let n=0;"complex64"!==e.dtype&&"string"!==e.dtype&&(n=e.size*M(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof ja||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;const t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,"string"===e.dtype&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),"complex64"!==e.dtype&&"string"!==e.dtype){const t=e.size*M(e.dtype);this.state.numBytes-=t}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(const e in this.state.registeredVariables){const t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),null!=this.state.registeredVariables[e.name]&&delete this.state.registeredVariables[e.name]}memory(){const e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,null==e.reasons&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;const t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(e=>e.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&0===this.state.kernelDepth}addTapeNode(e,t,n,r,a,s){const o={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:a},i=_r(e);null!=i&&(r=i.gradFunc),null!=r&&(o.gradient=e=>(e=e.map((e,t)=>{if(null==e){const e=n[t],r=j(e.size,e.dtype);return this.makeTensor(r,e.shape,e.dtype)}return e}),r(e.length>1?e:e[0],a,s))),this.state.activeTape.push(o)}keep(e){return e.kept=!0,e}startTape(){0===this.state.gradientDepth&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){const t={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){const t=us(e),n=new Set(t.map(e=>e.id));for(let a=0;a<this.state.activeScope.track.length;a++){const e=this.state.activeScope.track[a];e.kept||n.has(e.id)||e.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=0===this.state.scopeStack.length?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(e=>{e.kept||e.scopeId!==r.id||this.track(e)})}gradients(e,t,n,r=!1){if(d(t.length>0,()=>"gradients() received an empty list of xs."),null!=n&&"float32"!==n.dtype)throw new Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);const a=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));d(a instanceof Va,()=>"The result y returned by f() must be a tensor.");const s=function(e,t,n){const r={},a={};for(let u=0;u<t.length;u++)r[t[u].id]=!0;for(let u=0;u<e.length;u++){const n=e[u],s=n.inputs;for(const e in s){const o=s[e];let i=!1;for(let e=0;e<t.length;e++)if(r[o.id]){n.outputs.forEach(e=>r[e.id]=!0),i=!0,a[n.id]=!0;break}if(i)break}}const s={};s[n.id]=!0;const o={};for(let u=e.length-1;u>=0;u--){const t=e[u],n=t.inputs;for(let e=0;e<t.outputs.length;e++)if(s[t.outputs[e].id]){for(const e in n)s[n[e].id]=!0,o[t.id]=!0;break}}const i=[];for(let u=0;u<e.length;u++){const t=e[u];if(a[t.id]&&o[t.id]){const e={};for(const a in t.inputs){const n=t.inputs[a];r[n.id]&&(e[a]=n)}const n=Object.assign({},t);n.inputs=e,n.outputs=t.outputs,i.push(n)}}return i}(this.state.activeTape,t,a);if(!r&&0===s.length&&t.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const e={};e[a.id]=null==n?function(e){const t=U(f(e),"float32");return fs.makeTensor(t,e,"float32")}(a.shape):n,function(e,t,n,r){for(let a=t.length-1;a>=0;a--){const s=t[a],o=[];if(s.outputs.forEach(t=>{const n=e[t.id];null!=n?o.push(n):o.push(null)}),null==s.gradient)throw new Error(`Cannot compute gradient: gradient function not found for ${s.kernelName}.`);const i=s.gradient(o);for(const t in s.inputs){if(!(t in i))throw new Error(`Cannot backprop through input ${t}. Available gradients found: ${Object.keys(i)}.`);const a=n(()=>i[t]());if("float32"!==a.dtype)throw new Error(`Error in gradient for op ${s.kernelName}. The gradient of input ${t} must have 'float32' dtype, but has '${a.dtype}'`);const o=s.inputs[t];if(!y(a.shape,o.shape))throw new Error(`Error in gradient for op ${s.kernelName}. The gradient of input '${t}' has shape '${a.shape}', which does not match the shape of the input '${o.shape}'`);if(null==e[o.id])e[o.id]=a;else{const t=e[o.id];e[o.id]=r(t,a),t.dispose()}}}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(e,s,e=>this.tidy(e),gs);const r=t.map(t=>e[t.id]);return 0===this.state.gradientDepth&&(this.state.activeTape.forEach(e=>{for(const t of e.saved)t.dispose()}),this.state.activeTape=null),{value:a,grads:r}})}customGrad(e){return d(F(e),()=>"The f passed in customGrad(f) must be a function."),(...t)=>{let n;d(t.every(e=>e instanceof Va),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");const r={};t.forEach((e,t)=>{r[t]=e});return this.runKernelFunc({forwardFunc:(r,a)=>(n=e(...t,a),d(n.value instanceof Va,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),d(F(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),backwardsFunc:(e,r)=>{const a=n.gradFunc(e,r),s=Array.isArray(a)?a:[a];d(s.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),d(s.every(e=>e instanceof Va),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const o={};return s.forEach((e,t)=>{o[t]=()=>e}),o},inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){const t=Ta(),n=await this.backend.time(e);return n.wallMs=Ta()-t,n}track(e){return null!=this.state.activeScope&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new ds;for(const e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}function ms(){const e=te();if(null==e._tfengine){const t=new Z(e);e._tfengine=new hs(t)}var t;
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */return t=e._tfengine.ENV,ee=t,La=()=>e._tfengine,e._tfengine}hs.nextTensorId=0,hs.nextVariableId=0;const fs=ms();function gs(e,t){const n={a:e,b:t};return fs.runKernel(oe,n)}
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ys;function bs(e){if(void 0!==ys)return ys;if(e||"undefined"!=typeof navigator&&null!=navigator){if(e||(e=navigator),"ReactNative"===e.product)return!0;const t=e.userAgent||e.vendor||("undefined"!=typeof window?window.opera:"");if(!t){const t=e;return t.userAgentData&&t.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function ws(){return"undefined"!=typeof window&&null!=window.document||"undefined"!=typeof WorkerGlobalScope}const Ns=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:ws,isMobile:bs,mockIsMobile:function(e){ys=e}},Symbol.toStringTag,{value:"Module"})),xs=Y();
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function vs(e,t){let n=e;if(_a(e))return"string"===t?[]:[e.length];if(rs(e)){const t=e.channels||"RGBA";return[e.height,e.width*t.length]}if(as(e))return[e.buffer.size/(null==t?4:M(t))];if(!Array.isArray(e))return[];const r=[];for(;Array.isArray(n)||_a(n)&&"string"!==t;)r.push(n.length),n=n[0];return Array.isArray(e)&&Y().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Ts(e,r,[]),r}function Ts(e,t,n){if(n=n||[],!Array.isArray(e)&&!_a(e))return void d(0===t.length,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);d(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),d(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const r=t.slice(1);for(let a=0;a<e.length;++a)Ts(e[a],r,n.concat(a))}function Ss(e,t,n,r){if("string_or_numeric"!==e){if(null==e)throw new Error("Expected dtype cannot be null.");if("numeric"!==e&&e!==t||"numeric"===e&&"string"===t)throw new Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function ks(e,t,n,r="numeric"){if(e instanceof Ua())return Ss(r,e.dtype,t,n),e;let a=R(e);if("string"!==a&&["bool","int32","float32"].indexOf(r)>=0&&(a=r),Ss(r,a,t,n),null==e||!_a(e)&&!Array.isArray(e)&&"number"!=typeof e&&"boolean"!=typeof e&&"string"!=typeof e){const r=null==e?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${r}'`)}const s=vs(e,a);_a(e)||Array.isArray(e)||(e=[e]);const o="string"!==a?va(e,a):Ia(e,[],!0);return fs.makeTensor(o,s,a)}function Es(e,t,n,r="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((e,a)=>ks(e,`${t}[${a}]`,n,r))}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */xs.registerFlag("DEBUG",()=>!1,e=>{}),xs.registerFlag("IS_BROWSER",()=>ws()),xs.registerFlag("IS_NODE",()=>"undefined"!=typeof process&&void 0!==process.versions&&void 0!==process.versions.node),xs.registerFlag("IS_CHROME",()=>"undefined"!=typeof navigator&&null!=navigator&&null!=navigator.userAgent&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor)),xs.registerFlag("IS_SAFARI",()=>"undefined"!=typeof navigator&&null!=navigator&&null!=navigator.userAgent&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor)),xs.registerFlag("PROD",()=>!1),xs.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>xs.getBool("DEBUG")),xs.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0),xs.registerFlag("IS_TEST",()=>!1),xs.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>xs.getBool("DEBUG")),xs.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1),xs.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1),xs.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);const _s="__op";function Is(e){const t=Object.keys(e);if(1!==t.length)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const r=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n+=_s;const a=(...e)=>{fs.startScope(n);try{const t=r(...e);return H(t),fs.endScope(t),t}catch(t){throw fs.endScope(null),t}};return Object.defineProperty(a,"name",{value:n,configurable:!0}),a}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const As=Is({complex_:function(e,t){const n=ks(e,"real","complex"),r=ks(t,"imag","complex");h(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);const a={real:n,imag:r};return fs.runKernel(Me,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ms(e,t,n,r){if(null==r)r=R(e);else if("complex64"===r)throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(as(e)||rs(e)){if("float32"!==r&&"int32"!==r)throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return fs.backend.createTensorFromGPUData(e,t||n,r)}if(!_a(e)&&!Array.isArray(e)&&"number"!=typeof e&&"boolean"!=typeof e&&"string"!=typeof e)throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(null!=t){q(t);const e=f(t),r=f(n);d(e===r,()=>`Based on the provided shape, [${t}], the tensor should have ${e} values but has ${r}`);for(let a=0;a<n.length;++a){const e=n[a],r=a!==n.length-1||e!==f(t.slice(a));d(n[a]===t[a]||!r,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return _a(e)||Array.isArray(e)||(e=[e]),t=t||n,e="string"!==r?va(e,r):Ia(e,[],!0),fs.makeTensor(e,t,r)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $s(e,t,n){return Ms(e,t,vs(e,n),n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Os={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class Ds{static join(e){return new Ds(e).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,null==e)return;if(e instanceof Array||(e=[e]),0===(e=e.map(e=>_a(e)?e.buffer:e)).length)return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){const r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const a=t+r.byteLength;this.shards.push({buffer:r,start:t,end:a}),t=a}0===this.shards.length&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(0===this.shards.length)return new ArrayBuffer(0);if(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),(t=Math.min(this.byteLength,t))<=e)return new ArrayBuffer(0);const n=this.findShardForByte(e);if(-1===n)throw new Error(`Could not find start shard for byte ${e}`);const r=new ArrayBuffer(t-e),a=new Uint8Array(r);let s=0;for(let o=n;o<this.shards.length;o++){const n=this.shards[o],r=e+s-n.start,i=s,u=Math.min(t,n.end)-n.start,l=new Uint8Array(n.buffer,r,u-r);if(a.set(l,i),s+=l.length,t<n.end)break}return r}findShardForByte(e){if(0===this.shards.length||e<0||e>=this.byteLength)return-1;if(null!=this.bufferUniformSize)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(t){return e<t.start?-1:e>=t.end?1:0}if(0===t(this.shards[this.previousShardIndex]))return this.previousShardIndex;const n=function(e,t){let n=0,r=e.length;for(;n<=r;){const a=Math.floor((r-n)/2)+n,s=t(e[a]);if(0===s)return a;s<0?r=a:n=a+1}return-1}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(this.shards,t);return-1===n?-1:(this.previousShardIndex=n,this.previousShardIndex)}}function Cs(){Y().set("PROD",!0)}function Rs(){Y().set("DEBUG",!0)}function Fs(){Y().set("DEPRECATION_WARNINGS_ENABLED",!1)}function Bs(e){Y().getBool("DEPRECATION_WARNINGS_ENABLED")}function zs(){fs.disposeVariables()}function Ls(){return fs}function Ps(){return fs.memory()}function Vs(e){return fs.profile(e)}function Us(e,t){return fs.tidy(e,t)}function js(e){us(e).forEach(e=>e.dispose())}function Ws(e){return fs.keep(e)}function qs(e){return fs.time(e)}function Gs(e){return fs.setBackend(e)}function Ks(){return fs.ready()}function Hs(){return fs.backendName}function Js(e){fs.removeBackend(e)}function Zs(e){return fs.findBackend(e)}function Xs(e){return fs.findBackendFactory(e)}function Ys(e,t,n=1){return fs.registerBackend(e,t,n)}function Qs(){return fs.backend}function eo(e,t){Y().setPlatform(e,t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function to(e,t){const n=[],r=[],a=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);for(let s=0;s<a.length;++s){const o=a[s],i=Array.isArray(e)?e[s].tensor:e[o];if("float32"!==i.dtype&&"int32"!==i.dtype&&"bool"!==i.dtype&&"string"!==i.dtype&&"complex64"!==i.dtype)throw new Error(`Unsupported dtype in weight '${o}': ${i.dtype}`);const u={name:o,shape:i.shape,dtype:i.dtype};if("string"===i.dtype){const e=new Promise(async e=>{const t=await i.bytes(),n=t.reduce((e,t)=>e+t.length,0)+4*t.length,r=new Uint8Array(n);let a=0;for(let s=0;s<t.length;s++){const e=t[s],n=new Uint8Array(new Uint32Array([e.length]).buffer);r.set(n,a),a+=4,r.set(e,a),a+=e.length}e(r)});r.push(e)}else r.push(i.data());null!=t&&(u.group=t),n.push(u)}return{data:uo(await Promise.all(r)),specs:n}}function no(e,t){const n=new Ds(e),r={};let a=0;for(const s of t){const e=ro(s,(e,t)=>n.slice(a+e,a+t));r[s.name]=so(s,n.slice(a,a+e)),a+=e}return r}function ro(e,t){const n=f(e.shape);let r;if("quantization"in e){const t=e.quantization;r=Os[t.dtype]}else{if("string"===e.dtype){let e=0;for(let r=0;r<n;r++)e+=4+new Uint32Array(t(e,e+4))[0];return e}r=Os[e.dtype]}return n*r}async function ao(e,t){const n=f(e.shape);let r;if("quantization"in e){const t=e.quantization;r=Os[t.dtype]}else{if("string"===e.dtype){let e=0;for(let r=0;r<n;r++)e+=4+new Uint32Array(await t(e,e+4))[0];return e}r=Os[e.dtype]}return n*r}function so(e,t){const n=e.name,r=e.dtype,a=e.shape,s=f(a);let o,i=0;if("quantization"in e){const a=e.quantization;if("uint8"===a.dtype||"uint16"===a.dtype){if(!("min"in a)||!("scale"in a))throw new Error(`Weight ${e.name} with quantization ${a.dtype} doesn't have corresponding metadata min and scale.`)}else{if("float16"!==a.dtype)throw new Error(`Weight ${e.name} has unknown quantization dtype ${a.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);if("float32"!==r)throw new Error(`Weight ${e.name} is quantized with ${a.dtype} which only supports weights of type float32 not ${r}.`)}const u=Os[a.dtype],l="uint8"===a.dtype?new Uint8Array(t):new Uint16Array(t);if("float32"===r)if("uint8"===a.dtype||"uint16"===a.dtype){o=new Float32Array(l.length);for(let e=0;e<l.length;e++){const t=l[e];o[e]=t*a.scale+a.min}}else{if("float16"!==a.dtype)throw new Error(`Unsupported quantization type ${a.dtype} for weight type float32.`);{const e=function(){const e=function(){const e=e=>{let t=e<<13,n=0;for(;!(8388608&t);)n-=8388608,t<<=1;return t&=-8388609,n+=947912704,t|n},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}(),t=function(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}(),n=function(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}();return r=>{const a=new ArrayBuffer(4*r.length),s=new Uint32Array(a);for(let o=0;o<r.length;o++){const a=r[o],i=e[n[a>>10]+(1023&a)]+t[a>>10];s[o]=i}return new Float32Array(a)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */();o=e(l)}}else{if("int32"!==r)throw new Error(`Unsupported dtype in weight '${n}': ${r}`);if("uint8"!==a.dtype&&"uint16"!==a.dtype)throw new Error(`Unsupported quantization type ${a.dtype} for weight type int32.`);o=new Int32Array(l.length);for(let e=0;e<l.length;e++){const t=l[e];o[e]=Math.round(t*a.scale+a.min)}}i+=s*u}else if("string"===r){const n=f(e.shape);o=[];for(let e=0;e<n;e++){const e=new Uint32Array(t.slice(i,i+4))[0];i+=4;const n=new Uint8Array(t.slice(i,i+e));o.push(n),i+=e}}else{const e=Os[r];if("float32"===r)o=new Float32Array(t);else if("int32"===r)o=new Int32Array(t);else{if("bool"!==r){if("complex64"===r){o=new Float32Array(t);const e=new Float32Array(o.length/2),n=new Float32Array(o.length/2);for(let t=0;t<e.length;t++)e[t]=o[2*t],n[t]=o[2*t+1];const r=$s(e,a,"float32"),s=$s(n,a,"float32"),i=As(r,s);return r.dispose(),s.dispose(),i}throw new Error(`Unsupported dtype in weight '${n}': ${r}`)}o=new Uint8Array(t)}i+=s*e}return $s(o,a,r)}async function oo(e,t,n){let r=new Uint8Array(t);for(;r.byteLength<n;){const{done:t,value:a}=await e.read();if(t&&null==a){const e=n-r.byteLength;throw new Error(`Reader is done but ${e} bytes are still expected`)}const s=new Uint8Array(r.length+a.byteLength);s.set(r,0),s.set(new Uint8Array(a),r.length),r=s}return r.buffer}async function io(e,t){const n={},r=e.getReader();let a=new ArrayBuffer(0);for(const s of t){const e=await ao(s,async(e,t)=>(a=await oo(r,a,t),a.slice(e,t)));a=await oo(r,a,e);const t=a.slice(0,e);a=a.slice(e);const o=so(s,t);if(n[s.name]=o,"webgpu"===Hs()){const e=Qs();"uploadToGPU"in e&&f(o.shape)>=Y().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&e.uploadToGPU(o.dataId)}}return n}function uo(e){if(null===e)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(e=>{if(t+=e.byteLength,n.push(e.byteLength===e.buffer.byteLength?e:new e.constructor(e)),!(e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${e.constructor.name}`)});const r=new Uint8Array(t);let a=0;return n.forEach(e=>{r.set(new Uint8Array(e.buffer),a),a+=e.byteLength}),r.buffer}const lo="undefined"!=typeof Buffer&&("undefined"==typeof Blob||"undefined"==typeof atob||"undefined"==typeof btoa);function co(e){return lo?Buffer.byteLength(e,"utf8"):new Blob([e]).size}function po(e){return Ds.join(e)}function ho(e){for(e=e.trim();e.endsWith("/");)e=e.slice(0,e.length-1);const t=e.split("/");return t[t.length-1]}function mo(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return null!=e.signature&&(n.signature=e.signature),null!=e.userDefinedMetadata&&(n.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(n.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(n.initializerSignature=e.initializerSignature),null!=e.trainingConfig&&(n.trainingConfig=e.trainingConfig),n}function fo(e,t,n){const r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(null!=e.trainingConfig&&(r.trainingConfig=e.trainingConfig),null!=e.weightsManifest){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");r.weightSpecs=t,r.weightData=n}return null!=e.signature&&(r.signature=e.signature),null!=e.userDefinedMetadata&&(r.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(r.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(r.initializerSignature=e.initializerSignature),r}async function go(e,t){let n,r;return null!=e.weightsManifest&&([n,r]=await t(e.weightsManifest)),fo(e,n,r)}function yo(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:null==e.modelTopology?0:co(JSON.stringify(e.modelTopology)),weightSpecsBytes:null==e.weightSpecs?0:co(JSON.stringify(e.weightSpecs)),weightDataBytes:null==e.weightData?0:new Ds(e.weightData).byteLength}}function bo(e){const t=[];for(const n of e)t.push(...n.weights);return t}class wo{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return null==wo.instance&&(wo.instance=new wo),wo.instance}static registerSaveRouter(e){wo.getInstance().saveRouters.push(e)}static registerLoadRouter(e){wo.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return wo.getHandlers(e,"save")}static getLoadHandlers(e,t){return wo.getHandlers(e,"load",t)}static getHandlers(e,t,n){const r=[];return("load"===t?wo.getInstance().loadRouters:wo.getInstance().saveRouters).forEach(t=>{const a=t(e,n);null!==a&&r.push(a)}),r}}const No=e=>wo.getSaveHandlers(e),xo=(e,t)=>wo.getLoadHandlers(e,t),vo="tensorflowjs",To="models_store",So="model_info_store";function ko(){if(!Y().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e="undefined"==typeof window?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(null==t)throw new Error("The current browser does not appear to support IndexedDB.");return t}function Eo(e){const t=e.result;t.createObjectStore(To,{keyPath:"modelPath"}),t.createObjectStore(So,{keyPath:"modelPath"})}class _o{constructor(e){if(this.indexedDB=ko(),null==e||!e)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((e,n)=>{const r=this.indexedDB.open(vo,1);r.onupgradeneeded=()=>Eo(r),r.onsuccess=()=>{const a=r.result;if(null==t){const t=a.transaction(To,"readonly"),r=t.objectStore(To).get(this.modelPath);r.onsuccess=()=>{if(null==r.result)return a.close(),n(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));e(r.result.modelArtifacts)},r.onerror=e=>(a.close(),n(r.error)),t.oncomplete=()=>a.close()}else{t.weightData=Ds.join(t.weightData);const r=yo(t),o=a.transaction(So,"readwrite");let i,u,l=o.objectStore(So);try{i=l.put({modelPath:this.modelPath,modelArtifactsInfo:r})}catch(s){return n(s)}i.onsuccess=()=>{u=a.transaction(To,"readwrite");const i=u.objectStore(To);let c;try{c=i.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:r})}catch(s){return n(s)}c.onsuccess=()=>e({modelArtifactsInfo:r}),c.onerror=e=>{l=o.objectStore(So);const t=l.delete(this.modelPath);t.onsuccess=()=>(a.close(),n(c.error)),t.onerror=e=>(a.close(),n(c.error))}},i.onerror=e=>(a.close(),n(i.error)),o.oncomplete=()=>{null==u?a.close():u.oncomplete=()=>a.close()}}},r.onerror=e=>n(r.error)})}}_o.URL_SCHEME="indexeddb://";const Io=e=>{return Y().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(_o.URL_SCHEME)?(t=e.slice(_o.URL_SCHEME.length),new _o(t)):null;var t};wo.registerSaveRouter(Io),wo.registerLoadRouter(Io);class Ao{constructor(){this.indexedDB=ko()}async listModels(){return new Promise((e,t)=>{const n=this.indexedDB.open(vo,1);n.onupgradeneeded=()=>Eo(n),n.onsuccess=()=>{const r=n.result,a=r.transaction(So,"readonly"),s=a.objectStore(So).getAll();s.onsuccess=()=>{const t={};for(const e of s.result)t[e.modelPath]=e.modelArtifactsInfo;e(t)},s.onerror=e=>(r.close(),t(s.error)),a.oncomplete=()=>r.close()},n.onerror=e=>t(n.error)})}async removeModel(e){var t;return e=(t=e).startsWith(_o.URL_SCHEME)?t.slice(_o.URL_SCHEME.length):t,new Promise((t,n)=>{const r=this.indexedDB.open(vo,1);r.onupgradeneeded=()=>Eo(r),r.onsuccess=()=>{const a=r.result,s=a.transaction(So,"readwrite"),o=s.objectStore(So),i=o.get(e);let u;i.onsuccess=()=>{if(null==i.result)return a.close(),n(new Error(`Cannot find model with path '${e}' in IndexedDB.`));{const r=o.delete(e),s=()=>{u=a.transaction(To,"readwrite");const r=u.objectStore(To).delete(e);r.onsuccess=()=>t(i.result.modelArtifactsInfo),r.onerror=e=>n(i.error)};r.onsuccess=s,r.onerror=e=>(s(),a.close(),n(i.error))}},i.onerror=e=>(a.close(),n(i.error)),s.oncomplete=()=>{null==u?a.close():u.oncomplete=()=>a.close()}},r.onerror=e=>n(r.error)})}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mo="/",$o="tensorflowjs_models",Oo="info",Do="model_topology",Co="weight_specs",Ro="weight_data",Fo="model_metadata";function Bo(e){return{info:[$o,e,Oo].join(Mo),topology:[$o,e,Do].join(Mo),weightSpecs:[$o,e,Co].join(Mo),weightData:[$o,e,Ro].join(Mo),modelMetadata:[$o,e,Fo].join(Mo)}}function zo(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function Lo(e){const t=e.split(Mo);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(Mo)}class Po{constructor(e){if(!Y().getBool("IS_BROWSER")||"undefined"==typeof window||void 0===window.localStorage)throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,null==e||!e)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=Bo(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(e.modelTopology),r=JSON.stringify(e.weightSpecs),a=yo(e),s=Ds.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(a)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,r),this.LS.setItem(this.keys.weightData,function(e){if(lo)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let r=0,a=t.length;r<a;r++)n+=String.fromCharCode(t[r]);return btoa(n)}(s));const t={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:null!=e.signature?e.signature:void 0,userDefinedMetadata:null!=e.userDefinedMetadata?e.userDefinedMetadata:void 0,modelInitializer:null!=e.modelInitializer?e.modelInitializer:void 0,initializerSignature:null!=e.initializerSignature?e.initializerSignature:void 0,trainingConfig:null!=e.trainingConfig?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(t)),{modelArtifactsInfo:a}}catch(t){throw zo(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${a.modelTopologyBytes}, weightSpecsBytes=${a.weightSpecsBytes}, weightDataBytes=${a.weightDataBytes}.`)}}}async load(){const e=JSON.parse(this.LS.getItem(this.keys.info));if(null==e)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if("JSON"!==e.modelTopologyType)throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(null==n)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(null==r)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;const a=this.LS.getItem(this.keys.modelMetadata);if(null!=a){const e=JSON.parse(a);t.format=e.format,t.generatedBy=e.generatedBy,t.convertedBy=e.convertedBy,null!=e.signature&&(t.signature=e.signature),null!=e.userDefinedMetadata&&(t.userDefinedMetadata=e.userDefinedMetadata),null!=e.modelInitializer&&(t.modelInitializer=e.modelInitializer),null!=e.initializerSignature&&(t.initializerSignature=e.initializerSignature),null!=e.trainingConfig&&(t.trainingConfig=e.trainingConfig)}const s=this.LS.getItem(this.keys.weightData);if(null==s)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=function(e){if(lo){const t=Buffer.from(e,"base64");return t.buffer.slice(t.byteOffset,t.byteOffset+t.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let r=0;r<t.length;++r)n.set([t.charCodeAt(r)],r);return n.buffer}(s),t}}Po.URL_SCHEME="localstorage://";const Vo=e=>{return Y().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Po.URL_SCHEME)?(t=e.slice(Po.URL_SCHEME.length),new Po(t)):null;var t};wo.registerSaveRouter(Vo),wo.registerLoadRouter(Vo);class Uo{constructor(){d(Y().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),d("undefined"==typeof window||void 0!==window.localStorage,()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const e={},t=$o+Mo,n=Mo+Oo;for(let r=0;r<this.LS.length;++r){const a=this.LS.key(r);if(a.startsWith(t)&&a.endsWith(n)){e[Lo(a)]=JSON.parse(this.LS.getItem(a))}}return e}async removeModel(e){var t;const n=Bo(e=(t=e).startsWith(Po.URL_SCHEME)?t.slice(Po.URL_SCHEME.length):t);if(null==this.LS.getItem(n.info))throw new Error(`Cannot find model at path '${e}'`);const r=JSON.parse(this.LS.getItem(n.info));return zo(n),r}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jo="://";class Wo{constructor(){this.managers={}}static getInstance(){return null==Wo.instance&&(Wo.instance=new Wo),Wo.instance}static registerManager(e,t){d(null!=e,()=>"scheme must not be undefined or null."),e.endsWith(jo)&&(e=e.slice(0,e.indexOf(jo))),d(e.length>0,()=>"scheme must not be an empty string.");const n=Wo.getInstance();d(null==n.managers[e],()=>`A model store manager is already registered for scheme '${e}'.`),n.managers[e]=t}static getManager(e){const t=Wo.getInstance().managers[e];if(null==t)throw new Error(`Cannot find model manager for scheme '${e}'`);return t}static getSchemes(){return Object.keys(Wo.getInstance().managers)}}function qo(e){if(-1===e.indexOf(jo))throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${Wo.getSchemes().join(",")}`);return{scheme:e.split(jo)[0],path:e.split(jo)[1]}}async function Go(e,t,n=!1){d(e!==t,()=>`Old path and new path are the same: '${e}'`);const r=wo.getLoadHandlers(e);d(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),d(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);const a=r[0],s=wo.getSaveHandlers(t);d(s.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),d(s.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);const o=s[0],i=qo(e).scheme,u=qo(e).path,l=i===qo(e).scheme,c=await a.load();n&&l&&await Wo.getManager(i).removeModel(u);const p=await o.save(c);return n&&!l&&await Wo.getManager(i).removeModel(u),p.modelArtifactsInfo}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Ko{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if("utf-8"!==t&&"utf8"!==t)throw new Error(`Browser's encoder only supports utf-8, but got ${t}`);return null==this.textEncoder&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){"undefined"!=typeof window&&Y().getBool("USE_SETTIMEOUTCUSTOM")?(this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",e=>{if(e.source===window&&e.data.name===this.messageName){e.stopPropagation();(0,this.functionRefs[e.data.index])(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))):setTimeout(e,t)}isTypedArray(e){return Rr(e)}}if(Y().get("IS_BROWSER")){Y().setPlatform("browser",new Ko);try{Wo.registerManager(Po.URL_SCHEME,new Uo)}catch(Tf){}try{Wo.registerManager(_o.URL_SCHEME,new Ao)}catch(Tf){}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ho=()=>require("node-fetch");let Jo;class Zo{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return null!=Y().global.fetch?Y().global.fetch(e,t):(null==Jo&&(Jo=Ho()),Jo(e,t))}now(){const e=process.hrtime();return 1e3*e[0]+e[1]/1e6}encode(e,t){if("utf-8"!==t&&"utf8"!==t)throw new Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return 0===e.length?"":new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function Xo(e,t="float32",n){return t=t||"float32",q(e),new za(e,t,n)}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Y().get("IS_NODE")&&!Y().get("IS_BROWSER")&&Y().setPlatform("node",new Zo);const Yo=Is({cast_:function(e,t){const n=ks(e,"x","cast");if(!I(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if("string"===t&&"string"!==n.dtype||"string"!==t&&"string"===n.dtype)throw new Error("Only strings can be casted to strings");const r={x:n},a={dtype:t};return fs.runKernel(_e,r,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qo=Is({clone_:function(e){const t={x:ks(e,"x","clone","string_or_numeric")};return fs.runKernel(wt,t)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ei(e,t=!1){}
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ms();Pa={buffer:Xo,cast:Yo,clone:Qo,print:ei};const ti=Is({add_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t){let n=ks(e,"a","add"),r=ks(t,"b","add");[n,r]=ss(n,r);const a={a:n,b:r};return fs.runKernel(oe,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ni=Is({floorDiv_:function(e,t){let n=ks(e,"a","floorDiv"),r=ks(t,"b","floorDiv");[n,r]=ss(n,r);const a={a:n,b:r};return fs.runKernel(ht,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ri=Is({div_:function(e,t){let n=ks(e,"a","div"),r=ks(t,"b","div");if([n,r]=ss(n,r),"int32"===n.dtype&&"int32"===r.dtype)return ni(n,r);const a={a:n,b:r};return fs.runKernel(et,a,{})}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ai=Is({mul_:function(e,t){let n=ks(e,"a","mul"),r=ks(t,"b","mul");[n,r]=ss(n,r);const a={a:n,b:r};return fs.runKernel(Qt,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const si=Is({abs_:function(e){const t=ks(e,"x","abs");if("complex64"===t.dtype){const e={x:t};return fs.runKernel($e,e)}{const e={x:t};return fs.runKernel("Abs",e)}}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oi=Is({acos_:function(e){const t={x:ks(e,"x","acos")};return fs.runKernel(ae,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ii=Is({acosh_:function(e){const t={x:ks(e,"x","acosh")};return fs.runKernel(se,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ui=Is({addN_:function(e){d(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),d(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);const t=e.map((e,t)=>ks(e,`tensors${t}`,"addN")),n=t[0];t.forEach(e=>{if(e.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(e=>{if(!y(e.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const r=t;return fs.runKernel(ie,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const li=Is({all_:function(e,t=null,n=!1){const r={x:ks(e,"x","all","bool")},a={axis:t,keepDims:n};return fs.runKernel("All",r,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ci=Is({any_:function(e,t=null,n=!1){const r={x:ks(e,"x","any","bool")},a={axis:t,keepDims:n};return fs.runKernel("Any",r,a)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pi=Is({argMax_:function(e,t=0){const n={x:ks(e,"x","argMax")},r={axis:t};return fs.runKernel(ce,n,r)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const di=Is({argMin_:function(e,t=0){const n={x:ks(e,"x","argMin")},r={axis:t};return fs.runKernel(pe,n,r)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hi=Is({asin_:function(e){const t={x:ks(e,"x","asin")};return fs.runKernel(de,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mi=Is({asinh_:function(e){const t={x:ks(e,"x","asinh")};return fs.runKernel(he,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fi=Is({atan_:function(e){const t={x:ks(e,"x","atan")};return fs.runKernel(me,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gi=Is({atan2_:function(e,t){let n=ks(e,"a","atan2"),r=ks(t,"b","atan2");[n,r]=ss(n,r);const a={a:n,b:r};return fs.runKernel(ge,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yi=Is({atanh_:function(e){const t={x:ks(e,"x","atanh")};return fs.runKernel(fe,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bi(e,t,n,r,a="NHWC",s){return xi(e,[...t,e[3]],n,s,r,null,null,$i(a))}function wi(e,t,n,r,a,s,o="channelsLast"){const[i,u]=Si(t);let l;if("channelsLast"===o)l=[i,u,e[3],e[3]];else{if("channelsFirst"!==o)throw new Error(`Unknown dataFormat ${o}`);l=[i,u,e[1],e[1]]}return xi(e,l,n,r,a,s,!1,o)}function Ni(e,t,n,r,a,s,o="NDHWC"){const[i,u,l]=ki(t);let c,p;if("NDHWC"===o)p="channelsLast",c=[i,u,l,e[4],e[4]];else{if("NCDHW"!==o)throw new Error(`Unknown dataFormat ${o}`);p="channelsFirst",c=[i,u,l,e[1],e[1]]}return vi(e,c,n,r,a,!1,p,s)}function xi(e,t,n,r,a,s,o=!1,i="channelsLast"){let[u,l,c,p]=[-1,-1,-1,-1];if("channelsLast"===i)[u,l,c,p]=e;else{if("channelsFirst"!==i)throw new Error(`Unknown dataFormat ${i}`);[u,p,l,c]=e}const[d,h,,m]=t,[f,g]=Si(n),[y,b]=Si(r),w=Ei(d,y),N=Ei(h,b),{padInfo:x,outHeight:v,outWidth:T}=function(e,t,n,r,a,s,o,i,u){let l,c,p;if("number"==typeof e){l={top:e,bottom:e,left:e,right:e,type:0===e?"VALID":"NUMBER"};const a=function(e,t,n,r,a){null==r&&(r=Ti(e,t,n));const s=e[0],o=e[1],i=_i((s-t+2*r)/n+1,a),u=_i((o-t+2*r)/n+1,a);return[i,u]}([t,n],s,r,e,i);c=a[0],p=a[1]}else if("same"===e){c=Math.ceil(t/r),p=Math.ceil(n/a);const e=Math.max(0,(c-1)*r+s-t),i=Math.max(0,(p-1)*a+o-n),u=Math.floor(e/2),d=e-u,h=Math.floor(i/2);l={top:u,bottom:d,left:h,right:i-h,type:"SAME"}}else if("valid"===e)l={top:0,bottom:0,left:0,right:0,type:"VALID"},c=Math.ceil((t-s+1)/r),p=Math.ceil((n-o+1)/a);else{if("object"!=typeof e)throw Error(`Unknown padding parameter: ${e}`);{const d="channelsLast"===u?e[1][0]:e[2][0],h="channelsLast"===u?e[1][1]:e[2][1],m="channelsLast"===u?e[2][0]:e[3][0],f="channelsLast"===u?e[2][1]:e[3][1];l={top:d,bottom:h,left:m,right:f,type:0===d&&0===h&&0===m&&0===f?"VALID":"EXPLICIT"},c=_i((t-s+d+h)/r+1,i),p=_i((n-o+m+f)/a+1,i)}}return{padInfo:l,outHeight:c,outWidth:p}}(a,l,c,f,g,w,N,s,i),S=o?m*p:m;let k;return"channelsFirst"===i?k=[u,S,v,T]:"channelsLast"===i&&(k=[u,v,T,S]),{batchSize:u,dataFormat:i,inHeight:l,inWidth:c,inChannels:p,outHeight:v,outWidth:T,outChannels:S,padInfo:x,strideHeight:f,strideWidth:g,filterHeight:d,filterWidth:h,effectiveFilterHeight:w,effectiveFilterWidth:N,dilationHeight:y,dilationWidth:b,inShape:e,outShape:k,filterShape:t}}function vi(e,t,n,r,a,s=!1,o="channelsLast",i){let[u,l,c,p,d]=[-1,-1,-1,-1,-1];if("channelsLast"===o)[u,l,c,p,d]=e;else{if("channelsFirst"!==o)throw new Error(`Unknown dataFormat ${o}`);[u,d,l,c,p]=e}const[h,m,f,,g]=t,[y,b,w]=ki(n),[N,x,v]=ki(r),T=Ei(h,N),S=Ei(m,x),k=Ei(f,v),{padInfo:E,outDepth:_,outHeight:I,outWidth:A}=function(e,t,n,r,a,s,o,i,u,l,c){let p,d,h,m;"valid"===e&&(e=0);if("number"==typeof e){p={top:e,bottom:e,left:e,right:e,front:e,back:e,type:0===e?"VALID":"NUMBER"};const f=function(e,t,n,r,a,s){null==a&&(a=Ti(e,t[0],r[0]));const o=[0,0,0,n];for(let i=0;i<3;i++)e[i]+2*a>=t[i]&&(o[i]=_i((e[i]-t[i]+2*a)/r[i]+1,s));return o}([t,n,r,1],[i,u,l],1,[a,s,o],e,c);d=f[0],h=f[1],m=f[2]}else{if("same"!==e)throw Error(`Unknown padding parameter: ${e}`);{d=Math.ceil(t/a),h=Math.ceil(n/s),m=Math.ceil(r/o);const e=(d-1)*a+i-t,c=(h-1)*s+u-n,f=(m-1)*o+l-r,g=Math.floor(e/2),y=e-g,b=Math.floor(c/2),w=c-b,N=Math.floor(f/2);p={top:b,bottom:w,left:N,right:f-N,front:g,back:y,type:"SAME"}}}return{padInfo:p,outDepth:d,outHeight:h,outWidth:m}}(a,l,c,p,y,b,w,T,S,k,i),M=s?g*d:g;let $;return"channelsFirst"===o?$=[u,M,_,I,A]:"channelsLast"===o&&($=[u,_,I,A,M]),{batchSize:u,dataFormat:o,inDepth:l,inHeight:c,inWidth:p,inChannels:d,outDepth:_,outHeight:I,outWidth:A,outChannels:M,padInfo:E,strideDepth:y,strideHeight:b,strideWidth:w,filterDepth:h,filterHeight:m,filterWidth:f,effectiveFilterDepth:T,effectiveFilterHeight:S,effectiveFilterWidth:k,dilationDepth:N,dilationHeight:x,dilationWidth:v,inShape:e,outShape:$,filterShape:t}}function Ti(e,t,n,r=1){const a=Ei(t,r);return Math.floor((e[0]*(n-1)-n+a)/2)}function Si(e){return"number"==typeof e?[e,e,e]:2===e.length?[e[0],e[1],1]:e}function ki(e){return"number"==typeof e?[e,e,e]:e}function Ei(e,t){return t<=1?e:e+(e-1)*(t-1)}function _i(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function Ii(e){const[t,n,r]=Si(e);return 1===t&&1===n&&1===r}function Ai(e,t){return Ii(e)||Ii(t)}function Mi(e){return Si(e).every(e=>e>0)}function $i(e){if("NHWC"===e)return"channelsLast";if("NCHW"===e)return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function Oi(e,t,n){if(null!=n){if("string"==typeof t)throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if("number"==typeof t)d(b(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else{if("object"!=typeof t)throw Error(`Error in ${e}: Unknown padding parameter: ${t}`);t.forEach(t=>{t.forEach(t=>{d(b(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`)})})}}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Di=Is({reshape_:function(e,t){const n={x:ks(e,"x","reshape","string_or_numeric")},r={shape:t};return fs.runKernel(xn,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ci=Is({avgPool_:function(e,t,n,r,a){const s=ks(e,"x","avgPool","float32");d(Ai(n,1),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`);let o=s,i=!1;3===s.rank&&(i=!0,o=Di(s,[1,s.shape[0],s.shape[1],s.shape[2]])),d(4===o.rank,()=>`Error in avgPool: x must be rank 4 but got rank ${o.rank}.`),Oi("avgPool",r,a);const u={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:a};let c=fs.runKernel(ye,u,l);return c=Yo(c,s.dtype),i?Di(c,[c.shape[1],c.shape[2],c.shape[3]]):c}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ri=Is({avgPool3d_:function(e,t,n,r,a,s="NDHWC"){const o=ks(e,"x","avgPool3d","float32");let i=o,u=!1;4===o.rank&&(u=!0,i=Di(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),d(5===i.rank,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),d("NDHWC"===s,()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${s}`),d("number"==typeof n&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),Oi("avgPool3d",r,a);const l={x:i},c={filterSize:t,strides:n,pad:r,dimRoundingMode:a,dataFormat:s};let p=fs.runKernel(we,l,c);return p=Yo(p,i.dtype),u?Di(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fi=Is({concat_:function(e,t=0){d(e.length>=1,()=>"Pass at least one tensor to concat");const n=Es(e,"tensors","concat","string_or_numeric");if("complex64"===n[0].dtype&&n.forEach(e=>{if("complex64"!==e.dtype)throw new Error(`Cannot concatenate complex64 tensors with a tensor\n          with dtype ${e.dtype}. `)}),1===n.length)return Qo(n[0]);const r=n,a={axis:t};return fs.runKernel(Oe,r,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bi=Is({matMul_:function(e,t,n=!1,r=!1){let a=ks(e,"a","matMul"),s=ks(t,"b","matMul");[a,s]=ss(a,s);const o={a:a,b:s},i={transposeA:n,transposeB:r};return fs.runKernel(xe,o,i)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zi=Is({sigmoid_:function(e){const t={x:ks(e,"x","sigmoid","float32")};return fs.runKernel(Ln,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Li=Is({slice_:function(e,t,n){const r=ks(e,"x","slice","string_or_numeric");if(0===r.rank)throw new Error("Slicing scalar is not possible");const a={x:r},s={begin:t,size:n};return fs.runKernel(Rn,a,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pi=Is({tanh_:function(e){const t={x:ks(e,"x","tanh","float32")};return fs.runKernel(or,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vi=Is({basicLSTMCell_:function(e,t,n,r,a,s){const o=ks(e,"forgetBias","basicLSTMCell"),i=ks(t,"lstmKernel","basicLSTMCell"),u=ks(n,"lstmBias","basicLSTMCell"),l=ks(r,"data","basicLSTMCell"),c=ks(a,"c","basicLSTMCell"),p=ks(s,"h","basicLSTMCell"),d=Fi([l,p],1),h=Bi(d,i),m=ti(h,u),f=m.shape[0],g=m.shape[1]/4,y=[f,g],b=Li(m,[0,0],y),w=Li(m,[0,g],y),N=Li(m,[0,2*g],y),x=Li(m,[0,3*g],y),v=ti(ai(zi(b),Pi(w)),ai(c,zi(ti(o,N))));return[v,ai(Pi(v),zi(x))]}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ui=Is({batchToSpaceND_:function(e,t,n){const r=ks(e,"x","batchToSpaceND"),a=t.reduce((e,t)=>e*t);d(r.rank>=1+t.length,()=>`input rank is ${r.rank} but should be > than blockShape.length ${t.length}`),d(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),d(r.shape[0]%a===0,()=>`input tensor batch is ${r.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${a}`);const s={x:r},o={blockShape:t,crops:n};return fs.runKernel(ve,s,o)}});const ji=Is({batchNorm_:
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r,a,s){null==s&&(s=.001);const o=ks(e,"x","batchNorm"),i=ks(t,"mean","batchNorm"),u=ks(n,"variance","batchNorm");let l,c;null!=a&&(l=ks(a,"scale","batchNorm")),null!=r&&(c=ks(r,"offset","batchNorm")),d(i.rank===u.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),d(null==c||i.rank===c.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),d(null==l||i.rank===l.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const p={x:function(e){let t;return t=0===e.rank||1===e.rank?Di(e,[1,1,1,e.size]):2===e.rank?Di(e,[1,1,e.shape[0],e.shape[1]]):3===e.rank?Di(e,[1,e.shape[0],e.shape[1],e.shape[2]]):e,t}(o),scale:l,offset:c,mean:i,variance:u},h={varianceEpsilon:s},m=fs.runKernel(mt,p,h);return Di(m,o.shape)}});const Wi=Is({batchNorm2d_:function(e,t,n,r,a,s){const o=ks(e,"x","batchNorm"),i=ks(t,"mean","batchNorm"),u=ks(n,"variance","batchNorm");let l,c;return null!=a&&(l=ks(a,"scale","batchNorm")),null!=r&&(c=ks(r,"offset","batchNorm")),d(2===o.rank,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),d(2===i.rank||1===i.rank,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),d(2===u.rank||1===u.rank,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${u.rank}.`),null!=l&&d(2===l.rank||1===l.rank,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),null!=c&&d(2===c.rank||1===c.rank,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${c.rank}.`),ji(o,i,u,c,l,s)}});const qi=Is({batchNorm3d_:function(e,t,n,r,a,s){const o=ks(e,"x","batchNorm"),i=ks(t,"mean","batchNorm"),u=ks(n,"variance","batchNorm");let l,c;return null!=a&&(l=ks(a,"scale","batchNorm")),null!=r&&(c=ks(r,"offset","batchNorm")),d(3===o.rank,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),d(3===i.rank||1===i.rank,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),d(3===u.rank||1===u.rank,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${u.rank}.`),null!=l&&d(3===l.rank||1===l.rank,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),null!=c&&d(3===c.rank||1===c.rank,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${c.rank}.`),ji(o,i,u,c,l,s)}});const Gi=Is({batchNorm4d_:function(e,t,n,r,a,s){const o=ks(e,"x","batchNorm"),i=ks(t,"mean","batchNorm"),u=ks(n,"variance","batchNorm");let l,c;return null!=a&&(l=ks(a,"scale","batchNorm")),null!=r&&(c=ks(r,"offset","batchNorm")),d(4===o.rank,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),d(4===i.rank||1===i.rank,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),d(4===u.rank||1===u.rank,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${u.rank}.`),null!=l&&d(4===l.rank||1===l.rank,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),null!=c&&d(4===c.rank||1===c.rank,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${c.rank}.`),ji(o,i,u,c,l,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ki=Is({bincount_:function(e,t,n){const r=ks(e,"x","bincount"),a=ks(t,"weights","bincount");d("int32"===r.dtype,()=>`Error in bincount: input dtype must be int32, but got ${r.dtype}`),d(n>=0,()=>`size must be non-negative, but got ${n}.`),d(a.size===r.size||0===a.size,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${r.shape}, weights shape: ${a.shape}.`);const s={x:r,weights:a},o={size:n};return fs.runKernel(Te,s,o)}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hi=Is({bitwiseAnd_:function(e,t){const n=ks(e,"x","bitwiseAnd"),r=ks(t,"y","bitwiseAnd");if(!y(n.shape,r.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${r.shape}`);if("int32"!==n.dtype||"int32"!==r.dtype)throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${r.dtype}`);const a={a:n,b:r};return fs.runKernel(Se,a)}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ji=Is({broadcastArgs_:function(e,t){const n=ks(e,"s0","broadcastArgs","int32"),r=ks(t,"s1","broadcastArgs","int32");if(1!==n.rank)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(1!==r.rank)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${r.rank}`);const a={s0:n,s1:r};return fs.runKernel(Ee,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zi=Is({broadcastTo_:function(e,t){let n=ks(e,"broadcastTo","x");const r=n.shape;if(q(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const e=n.shape.slice();for(;e.length<t.length;)e.unshift(1);n=Di(n,e)}const a=n.shape,s=Array.from(t);for(let u=t.length-1;u>=0;u--)if(a[u]===t[u])s[u]=1;else if(1!==n.shape[u])throw new Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(0===s.map((e,t)=>e>1?t:-1).filter(e=>e>=0).length)return Qo(n);const o={x:n},i={reps:s};return fs.runKernel(ir,o,i)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xi=Is({ceil_:function(e){const t={x:ks(e,"x","ceil","float32")};return fs.runKernel(Ie,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yi(e,t,n){q(e);const r={shape:e,value:t,dtype:n=n||R(t)};return fs.runKernel(ct,{},r)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qi=Is({clipByValue_:function(e,t,n){const r=ks(e,"x","clipByValue");if(d(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return Yi(r.shape,t,r.dtype);const a={x:r},s={clipValueMin:t,clipValueMax:n};return fs.runKernel(Ae,a,s)}});const eu=Is({concat1d_:function(e){return Fi(e,0)}});const tu=Is({concat2d_:function(e,t){return Fi(e,t)}});const nu=Is({concat3d_:function(e,t){return Fi(e,t)}});const ru=Is({concat4d_:function(e,t){return Fi(e,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const au=Is({conv2d_:function(e,t,n,r,a="NHWC",s=[1,1],o){const i=ks(e,"x","conv2d","float32"),u=ks(t,"filter","conv2d","float32");let l=i,c=!1;3===i.rank&&(c=!0,l=Di(i,[1,i.shape[0],i.shape[1],i.shape[2]])),d(4===l.rank,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),d(4===u.rank,()=>`Error in conv2d: filter must be rank 4, but got rank ${u.rank}.`),Oi("conv2d",r,o);const p="NHWC"===a?l.shape[3]:l.shape[1];d(p===u.shape[2],()=>`Error in conv2d: depth of input (${p}) must match input depth for filter ${u.shape[2]}.`),d(Ai(n,s),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${s}'`),d(Mi(s),()=>"Error in conv2D: Dilated rates should be larger than 0."),d(Mi(n),()=>"Error in conv2D: Strides should be larger than 0.");const h={x:l,filter:u},m={strides:n,pad:r,dataFormat:a,dilations:s,dimRoundingMode:o},f=fs.runKernel(De,h,m);return c?Di(f,[f.shape[1],f.shape[2],f.shape[3]]):f}});const su=Is({conv1d_:function(e,t,n,r,a="NWC",s=1,o){const i=ks(e,"x","conv1d"),u=ks(t,"filter","conv1d");let l=i,c=!1;2===i.rank&&(c=!0,l=Di(i,[1,i.shape[0],i.shape[1]])),d(3===l.rank,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),d(3===u.rank,()=>`Error in conv1d: filter must be rank 3, but got rank ${u.rank}.`),Oi("conv1d",r,o),d(l.shape[2]===u.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${u.shape[1]}.`),d(Ai(n,s),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${s}'`),d(Mi(s),()=>"Error in conv1D: Dilated rates should be larger than 0."),d(Mi(n),()=>"Error in conv1D: Stride should be larger than 0."),d("NWC"===a,()=>`Error in conv1d: got dataFormat of ${a} but only NWC is currently supported.`);const p=Di(u,[1,u.shape[0],u.shape[1],u.shape[2]]),h=Di(l,[l.shape[0],1,l.shape[1],l.shape[2]]),m=au(h,p,[1,n],r,"NHWC",[1,s],o);return Di(m,c?[m.shape[2],m.shape[3]]:[m.shape[0],m.shape[2],m.shape[3]])}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ou=Is({conv2DBackpropInput_:function(e,t,n,r,a,s="NHWC",o){d(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let i=e,u=t,l=!1;3===t.rank&&(l=!0,u=Di(t,[1,t.shape[0],t.shape[1],t.shape[2]]),i=[1,e[0],e[1],e[2]]),d(4===i.length,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),d(4===u.rank,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${u.rank}`),d(4===n.rank,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const c="NHWC"===s?i[3]:i[1],p="NHWC"===s?u.shape[3]:u.shape[1];d(c===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${c}) must match input depth for filter ${n.shape[2]}.`),d(p===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${p}) must match output depth for filter ${n.shape[3]}.`),Oi("conv2dDerInput",a,o);const h={dy:u,filter:n},m={strides:r,pad:a,dataFormat:s,dimRoundingMode:o,inputShape:i},f=fs.runKernel(Re,h,m);return l?Di(f,[f.shape[1],f.shape[2],f.shape[3]]):f}});const iu=Is({conv2dTranspose_:function(e,t,n,r,a,s){const o=ks(e,"x","conv2dTranspose"),i=ks(t,"filter","conv2dTranspose");return ou(n,o,i,r,a,"NHWC",s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uu=Is({conv3d_:function(e,t,n,r,a="NDHWC",s=[1,1,1]){const o=ks(e,"x","conv3d"),i=ks(t,"filter","conv3d");let u=o,l=!1;4===o.rank&&(l=!0,u=Di(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),d(5===u.rank,()=>`Error in conv3d: input must be rank 5, but got rank ${u.rank}.`),d(5===i.rank,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),d(u.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${u.shape[4]}) must match input depth for filter ${i.shape[3]}.`),d(Ai(n,s),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${s}'`),d("NDHWC"===a,()=>`Error in conv3d: got dataFormat of ${a} but only NDHWC is currently supported.`),d(Mi(s),()=>"Error in conv3D: Dilated rates should be larger than 0."),d(Mi(n),()=>"Error in conv3D: Strides should be larger than 0.");const c={x:u,filter:i},p={strides:n,pad:r,dataFormat:a,dilations:s},h=fs.runKernel(Fe,c,p);return l?Di(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lu=Is({conv3DBackpropInput_:function(e,t,n,r,a){d(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let s=e,o=t,i=!1;4===t.rank&&(i=!0,o=Di(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),s=[1,e[0],e[1],e[2],e[3]]);const u=s[4],l=o.shape[4];d(5===s.length,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${s.length}.`),d(5===o.rank,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),d(5===n.rank,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),d(u===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[3]}.`),d(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);const c={dy:o,filter:n},p={pad:a,strides:r,inputShape:s},h=fs.runKernel(ze,c,p);return i?Di(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}});const cu=Is({conv3dTranspose_:function(e,t,n,r,a){const s=ks(e,"x","conv3dTranspose"),o=ks(t,"filter","conv3dTranspose");return lu(n,s,o,r,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pu=Is({cos_:function(e){const t={x:ks(e,"x","cos","float32")};return fs.runKernel("Cos",t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const du=Is({cosh_:function(e){const t={x:ks(e,"x","cosh","float32")};return fs.runKernel(Pe,t)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hu=Is({cumprod_:function(e,t=0,n=!1,r=!1){const a={x:ks(e,"x","cumprod")},s={axis:t,exclusive:n,reverse:r};return fs.runKernel(Ve,a,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mu=Is({cumsum_:function(e,t=0,n=!1,r=!1){const a={x:ks(e,"x","cumsum")},s={axis:t,exclusive:n,reverse:r};return fs.runKernel(Ue,a,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fu=Is({denseBincount_:function(e,t,n,r=!1){const a=ks(e,"x","denseBincount"),s=ks(t,"weights","denseBincount");d("int32"===a.dtype,()=>`Error in denseBincount: input dtype must be int32, but got ${a.dtype}`),d(a.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${a.rank}.`),d(n>=0,()=>`size must be non-negative, but got ${n}.`),d(s.size===a.size||0===s.size,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${a.shape}, weights shape: ${s.shape}.`);const o={x:a,weights:s},i={size:n,binaryOutput:r};return fs.runKernel(We,o,i)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gu=Is({depthToSpace_:function(e,t,n="NHWC"){const r=ks(e,"x","depthToSpace","float32"),a="NHWC"===n?r.shape[1]:r.shape[2],s="NHWC"===n?r.shape[2]:r.shape[3],o="NHWC"===n?r.shape[3]:r.shape[1];d(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),d(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying\n    ${a} and ${t}  for depthToSpace with input shape\n    ${r.shape}`),d(s*t>=0,()=>`Negative dimension size caused by overflow when multiplying\n    ${s} and ${t} for depthToSpace with input shape\n        ${r.shape}`),d(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${r.shape}`);const i={x:r},u={blockSize:t,dataFormat:n};return fs.runKernel(qe,i,u)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yu=Is({depthwiseConv2d_:function(e,t,n,r,a="NHWC",s=[1,1],o){const i=ks(e,"x","depthwiseConv2d","float32"),u=ks(t,"filter","depthwiseConv2d","float32");let l=i,c=!1;3===i.rank&&(c=!0,l=Di(i,[1,i.shape[0],i.shape[1],i.shape[2]])),d(4===l.rank,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),d(4===u.rank,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${u.rank}.`);const p="NHWC"===a?l.shape[3]:l.shape[1];d(p===u.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${p}) must match the inChannels dimension in filter ${u.shape[2]}.`),Oi("depthwiseConv2d",r,o);const h={x:l,filter:u},m={strides:n,pad:r,dataFormat:a,dilations:s,dimRoundingMode:o},f=fs.runKernel(Ge,h,m);return c?Di(f,[f.shape[1],f.shape[2],f.shape[3]]):f}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bu=Is({diag_:function(e){const t={x:ks(e,"x","diag")};return fs.runKernel(Je,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wu=Is({dilation2d_:function(e,t,n,r,a=[1,1],s="NHWC"){const o=ks(e,"x","dilation2d"),i=ks(t,"filter","dilation2d");d(3===o.rank||4===o.rank,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),d(3===i.rank,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),d("NHWC"===s,()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${s}`);let u=o,l=!1;3===o.rank&&(u=Di(o,[1,o.shape[0],o.shape[1],o.shape[2]]),l=!0),d(u.shape[3]===i.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${u.shape[3]} vs ${i.shape[2]}`);const c={x:u,filter:i},p={strides:n,pad:r,dilations:a},h=fs.runKernel(Ze,c,p);return l?Di(h,[h.shape[1],h.shape[2],h.shape[3]]):h}});
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nu(e,t){const n=e.length,r=[];for(let a=0;a<n;a++){const s=n-1-a,o=e[s]||1;(t[t.length-1-a]||1)>1&&1===o&&r.unshift(s)}return r}function xu(e,t){const n=[];for(let r=0;r<t.length;r++){const a=e[e.length-r-1],s=t.length-r-1,o=t[s];(null==a||1===a&&o>1)&&n.unshift(s)}return n}function vu(e,t){const n=Math.max(e.length,t.length),r=new Array(n);for(let a=0;a<n;a++){let s=e[e.length-a-1];null==s&&(s=1);let o=t[t.length-a-1];if(null==o&&(o=1),1===s)r[n-a-1]=o;else if(1===o)r[n-a-1]=s;else{if(s!==o){throw Error(`Operands could not be broadcast together with shapes ${e} and ${t}.`)}r[n-a-1]=s}}return r}const Tu=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:vu,getBroadcastDims:Nu,getReductionAxes:xu},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Su=Is({equal_:function(e,t){let n=ks(e,"a","equal","string_or_numeric"),r=ks(t,"b","equal","string_or_numeric");[n,r]=ss(n,r),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(st,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ku=Is({where_:function(e,t,n){const r=ks(t,"a","where"),a=ks(n,"b","where"),s=ks(e,"condition","where","bool"),o=vu(vu(s.shape,r.shape),a.shape),i={condition:Zi(s,o),t:Zi(r,o),e:Zi(a,o)};return fs.runKernel(Dn,i)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Eu=Is({zerosLike_:function(e){const t={x:ks(e,"x","zerosLike")};return fs.runKernel(fr,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _u=Is({divNoNan_:function(e,t){let n=ks(e,"a","div"),r=ks(t,"b","div");[n,r]=ss(n,r);const a=ri(n,r),s=Eu(a),o=Su(r,s);return ku(o,s,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Iu=Is({dot_:function(e,t){const n=ks(e,"t1","dot"),r=ks(t,"t2","dot");d(!(1!==n.rank&&2!==n.rank||1!==r.rank&&2!==r.rank),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${r.rank}.`);const a=1===n.rank?n.size:n.shape[1],s=1===r.rank?r.size:r.shape[0];if(d(a===s,()=>`Error in dot: inner dimensions of inputs must match, but got ${a} and ${s}.`),1===n.rank&&1===r.rank){const e=Di(n,[1,-1]),t=Di(r,[-1,1]),a=Bi(e,t);return Di(a,[])}if(1===n.rank&&2===r.rank){const e=Di(n,[1,-1]),t=Di(r,[r.shape[0],r.shape[1]]),a=Bi(e,t);return Di(a,[a.size])}if(2===n.rank&&1===r.rank){const e=Di(r,[-1,1]),t=Bi(n,e);return Di(t,[t.size])}{const e=Di(r,[r.shape[0],r.shape[1]]);return Bi(n,e)}}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Au=Is({einsum_:function(e,...t){const n=t.map((e,t)=>ks(e,`tensors${t}`,"einsum")),r={equation:e};return fs.runKernel(tt,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mu=Is({elu_:function(e){const t={x:ks(e,"x","elu","float32")};return fs.runKernel("Elu",t)}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $u=Is({ensureShape_:function(e,t){const n=ks(e,"x","ensureShape","string_or_numeric");if(!g(n.shape,t))throw new Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${t}`);return e}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ou=Is({erf_:function(e){let t=ks(e,"x","erf");d("int32"===t.dtype||"float32"===t.dtype,()=>"Input dtype must be `int32` or `float32`."),"int32"===t.dtype&&(t=Yo(t,"float32"));const n={x:t};return fs.runKernel("Erf",n)}});
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Du(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function Cu(e,t,n){const r=e.length+t.length,a=[];let s=0,o=0;for(let i=0;i<r;i++)-1===n.indexOf(i)?a.push(e[s++]):a.push(t[o++]);return a}function Ru(e,t){const n=[],r=e.length;for(let a=0;a<r;a++)-1===t.indexOf(a)&&n.push(e[a]);return[n,t.map(t=>e[t])]}function Fu(e,t){return Cu(e,t.map(e=>1),t)}function Bu(e,t,n){d(Du(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function zu(e,t){if(Du(e,t))return null;const n=[];for(let r=0;r<t;++r)-1===e.indexOf(r)&&n.push(r);return e.forEach(e=>n.push(e)),n}function Lu(e){return e.map((e,t)=>[t,e]).sort((e,t)=>e[1]-t[1]).map(e=>e[0])}function Pu(e,t){const n=[];for(let r=t-e;r<t;++r)n.push(r);return n}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vu=Is({max_:function(e,t=null,n=!1){const r={x:ks(e,"x","max")},a={reductionIndices:t,keepDims:n};return fs.runKernel("Max",r,a)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uu=Is({min_:function(e,t=null,n=!1){const r={x:ks(e,"x","min")},a={axis:t,keepDims:n};return fs.runKernel("Min",r,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ju=Is({pow_:function(e,t){let n=ks(e,"base","pow"),r=ks(t,"exp","pow");[n,r]=ss(n,r);const a={a:n,b:r};return fs.runKernel("Pow",a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wu(e,t){if((_a(e)&&"string"!==t||Array.isArray(e))&&"complex64"!==t)throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if("string"===t&&_a(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Ms(e,[],[],t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qu=Is({sqrt_:function(e){const t={x:ks(e,"x","sqrt","float32")};return fs.runKernel(Vn,t)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gu=Is({square_:function(e){const t=ks(e,"x","square");return fs.runKernel("Square",{x:t},{})}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ku=Is({sum_:function(e,t=null,n=!1){let r=ks(e,"x","sum");"bool"===r.dtype&&(r=Yo(r,"int32"));const a={x:r},s={axis:t,keepDims:n};return fs.runKernel("Sum",a,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hu(e,t,n=null){if(0===e.rank)return si(e);if(1!==e.rank&&null===n)return Hu(Di(e,[-1]),t,n);if(1===e.rank||"number"==typeof n||Array.isArray(n)&&1===n.length){if(1===t)return Ku(si(e),n);if(t===1/0)return Vu(si(e),n);if(t===-1/0)return Uu(si(e),n);if("euclidean"===t||2===t)return qu(Ku(ju(si(e),Wu(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&2===n.length){if(1===t)return Vu(Ku(si(e),n[0]),n[1]-1);if(t===1/0)return Vu(Ku(si(e),n[1]),n[0]);if(t===-1/0)return Uu(Ku(si(e),n[1]),n[0]);if("fro"===t||"euclidean"===t)return qu(Ku(Gu(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const Ju=Is({norm_:function(e,t="euclidean",n=null,r=!1){const a=Hu(e=ks(e,"x","norm"),t,n);let s=a.shape;if(r){const t=T(n,e.shape);s=Fu(a.shape,t)}return Di(a,s)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zu=Is({euclideanNorm_:function(e,t=null,n=!1){return Ju(e,"euclidean",t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xu=Is({exp_:function(e){const t={x:ks(e,"x","exp")};return fs.runKernel("Exp",t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yu=Is({expandDims_:function(e,t=0){const n=ks(e,"x","expandDims","string_or_numeric");d(t<=n.rank,()=>"Axis must be <= rank of the tensor");const r={input:n},a={dim:t};return fs.runKernel(it,r,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qu=Is({expm1_:function(e){const t={x:ks(e,"x","expm1")};return fs.runKernel(ut,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const el=Is({tile_:function(e,t){const n=ks(e,"x","tile","string_or_numeric");d(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const r={x:n},a={reps:t};return fs.runKernel(ir,r,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tl=Is({eye_:function(e,t,n,r="float32"){null==t&&(t=e);const a=Xo([e,t],r),s=e<=t?e:t;for(let i=0;i<s;++i)a.set(1,i,i);const o=Di(a.toTensor(),[e,t]);if(null==n)return o;if(1===n.length)return el(Yu(o,0),[n[0],1,1]);if(2===n.length)return el(Yu(Yu(o,0),0),[n[0],n[1],1,1]);if(3===n.length)return el(Yu(Yu(Yu(o,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nl=Is({floor_:function(e){const t={x:ks(e,"x","floor","float32")};return fs.runKernel(dt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rl=Is({gather_:function(e,t,n=0,r=0){const a={x:ks(e,"x","gather"),indices:ks(t,"indices","gather","int32")},s={axis:n,batchDims:r};return fs.runKernel(ft,a,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const al=Is({greater_:function(e,t){let n=ks(e,"a","greater","string_or_numeric"),r=ks(t,"b","greater","string_or_numeric");[n,r]=ss(n,r),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(yt,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sl=Is({greaterEqual_:function(e,t){let n=ks(e,"a","greaterEqual","string_or_numeric"),r=ks(t,"b","greaterEqual","string_or_numeric");[n,r]=ss(n,r),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(bt,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ol=Is({imag_:function(e){const t={input:ks(e,"input","imag")};return fs.runKernel(xt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const il=Is({isFinite_:function(e){const t={x:ks(e,"x","isFinite")};return fs.runKernel(vt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ul=Is({isInf_:function(e){const t={x:ks(e,"x","isInf")};return fs.runKernel(Tt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ll=Is({isNaN_:function(e){const t={x:ks(e,"x","isNaN")};return fs.runKernel(St,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cl=Is({leakyRelu_:function(e,t=.2){const n={x:ks(e,"x","leakyRelu")},r={alpha:t};return fs.runKernel(kt,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pl=Is({less_:function(e,t){let n=ks(e,"a","less","string_or_numeric"),r=ks(t,"b","less","string_or_numeric");[n,r]=ss(n,r),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(Et,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dl=Is({lessEqual_:function(e,t){let n=ks(e,"a","lessEqual","string_or_numeric"),r=ks(t,"b","lessEqual","string_or_numeric");[n,r]=ss(n,r),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(_t,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hl(e,t,n){if(n<=0)throw new Error("The number of values should be positive.");const r={start:e,stop:t,num:n};return fs.runKernel(It,{},r)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ml=Is({localResponseNormalization_:function(e,t=5,n=1,r=1,a=.5){const s=ks(e,"x","localResponseNormalization");d(4===s.rank||3===s.rank,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got\n               rank ${s.rank}.`),d(b(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=s,i=!1;3===s.rank&&(i=!0,o=Di(s,[1,s.shape[0],s.shape[1],s.shape[2]]));const u={x:o},l={depthRadius:t,bias:n,alpha:r,beta:a},c=fs.runKernel("LRN",u,l);return i?Di(c,[c.shape[1],c.shape[2],c.shape[3]]):c}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fl=Is({log_:function(e){const t={x:ks(e,"x","log","float32")};return fs.runKernel("Log",t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gl=Is({log1p_:function(e){const t={x:ks(e,"x","log1p")};return fs.runKernel(Mt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yl(e){return d(F(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{const r=ks(t,"x","tf.grad","string_or_numeric"),a=null!=n?ks(n,"dy","tf.grad"):null;return fs.tidy(()=>{const{value:t,grads:n}=fs.gradients(()=>e(r),[r],a);return null!=a&&h(t.shape,a.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),Tl(n),n[0]})}}function bl(e){return d(F(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{d(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const r=Es(t,"args","tf.grads","string_or_numeric"),a=null!=n?ks(n,"dy","tf.grads"):null;return fs.tidy(()=>{const{value:t,grads:n}=fs.gradients(()=>e(...r),r,a);return null!=a&&h(t.shape,a.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Tl(n),n})}}function wl(e){return d(F(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{d(t instanceof Va,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),d(null==n||n instanceof Va,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:r,value:a}=fs.gradients(()=>e(t),[t],n);return Tl(r),{grad:r[0],value:a}}}function Nl(e){return d(F(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{d(Array.isArray(t)&&t.every(e=>e instanceof Va),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),d(null==n||n instanceof Va,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const r=fs.gradients(()=>e(...t),t,n);return null!=n&&h(r.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Tl(r.grads),r}}function xl(e,t){d(F(e),()=>"The f passed in variableGrads(f) must be a function"),d(null==t||Array.isArray(t)&&t.every(e=>e instanceof ja),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=null!=t;if(!n){t=[];for(const e in fs.registeredVariables)t.push(fs.registeredVariables[e])}const r=n?t.filter(e=>!e.trainable):null,a=t.length;d((t=t.filter(e=>e.trainable)).length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${a} variables is trainable.`);const{value:s,grads:o}=fs.gradients(e,t,null,!0);d(o.some(e=>null!=e),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),d(0===s.rank,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${s.rank} tensor`);const i={};return t.forEach((e,t)=>{null!=o[t]&&(i[e.name]=o[t])}),null!=r&&r.forEach(e=>i[e.name]=null),{value:s,grads:i}}function vl(e){return fs.customGrad(e)}function Tl(e){if(e.filter(e=>null==e).length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that\n    the f you passed encloses all operations that lead from x to y.")}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sl=Is({neg_:function(e){const t={x:ks(e,"x","neg")};return fs.runKernel("Neg",t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kl=Is({softplus_:function(e){const t={x:ks(e,"x","softplus")};return fs.runKernel(Pn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const El=Is({logSigmoid_:function(e){const t=ks(e,"x","logSigmoid");return vl(e=>({value:Sl(kl(Sl(e))),gradFunc:t=>ai(t,zi(Sl(e)))}))(t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _l=Is({sub_:function(e,t){let n=ks(e,"a","sub"),r=ks(t,"b","sub");[n,r]=ss(n,r);const a={a:n,b:r};return fs.runKernel("Sub",a)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Il=Is({logSoftmax_:function(e,t=-1){const n=ks(e,"logits","logSoftmax");if(-1===t&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return vl((e,n)=>{const r=Vu(e,t,!0),a=_l(e,r),s=_l(Yo(a,"float32"),fl(Ku(Xu(a),t,!0)));n([s]);return{value:s,gradFunc:(e,n)=>{const[r]=n,a=Xu(r);return _l(e,ai(Ku(e,t,!0),a))}}})(n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Al=Is({logSumExp_:function(e,t=null,n=!1){const r=ks(e,"x","logSumExp"),a=T(t,r.shape),s=Vu(r,a,!0),o=_l(r,s),i=Xu(o),u=Ku(i,a),l=fl(u),c=ti(Di(s,l.shape),l);if(n){const e=Fu(c.shape,a);return Di(c,e)}return c}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ml=Is({logicalAnd_:function(e,t){const n=ks(e,"a","logicalAnd","bool"),r=ks(t,"b","logicalAnd","bool");vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel($t,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $l=Is({logicalNot_:function(e){const t={x:ks(e,"x","logicalNot","bool")};return fs.runKernel(Ot,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ol=Is({logicalOr_:function(e,t){const n=ks(e,"a","logicalOr","bool"),r=ks(t,"b","logicalOr","bool");vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(Dt,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dl=Is({logicalXor_:function(e,t){const n=ks(e,"a","logicalXor","bool"),r=ks(t,"b","logicalXor","bool");return vu(n.shape,r.shape),Ml(Ol(e,t),$l(Ml(e,t)))}}),Cl=2147483648;
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rl=Is({searchSorted_:function(e,t,n="left"){const r=ks(e,"sortedSequence","searchSorted"),a=ks(t,"values","searchSorted"),s=r.shape[r.shape.length-1],o=a.shape[a.shape.length-1],i=Di(r,[-1,s]),u=Di(a,[-1,o]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==u.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(f(u.shape)>=Cl)throw new Error("values tensor size must less than 2147483648");if(i.shape[1]>=Cl)throw new Error(`trailing dim_size must less than 2147483648 for int32 output type, was ${i.shape[1]}`);const l={sortedSequence:i,values:u},c={side:n};return fs.runKernel(On,l,c)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fl(e,t){return Rl(e,t,"left")}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bl=Is({maxPool_:function(e,t,n,r,a){const s=ks(e,"x","maxPool");let o=s,i=!1;3===s.rank&&(i=!0,o=Di(s,[1,s.shape[0],s.shape[1],s.shape[2]])),d(4===o.rank,()=>`Error in maxPool: input must be rank 4 but got rank ${o.rank}.`),d(Ai(n,1),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '1'`),Oi("maxPool",r,a);const u={x:o},l={filterSize:t,strides:n,pad:r,dimRoundingMode:a},c=fs.runKernel(Ut,u,l);return i?Di(c,[c.shape[1],c.shape[2],c.shape[3]]):c}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zl=Is({maxPool3d_:function(e,t=[1,1,1],n,r,a,s="NDHWC"){const o=ks(e,"x","maxPool3d");let i=o,u=!1;4===o.rank&&(u=!0,i=Di(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),d(5===i.rank,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),d("NDHWC"===s,()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${s}`),Oi("maxPool3d",r,a);const l={x:i},c={filterSize:t,strides:n,pad:r,dimRoundingMode:a,dataFormat:s},p=fs.runKernel(Wt,l,c);return u?Di(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ll=Is({maxPoolWithArgmax_:function(e,t,n,r,a=!1){const s={x:ks(e,"x","maxPoolWithArgmax")},o={filterSize:t,strides:n,pad:r,includeBatchInIndex:a},i=fs.runKernel(Gt,s,o);return{result:i[0],indexes:i[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pl=Is({maximum_:function(e,t){let n=ks(e,"a","maximum"),r=ks(t,"b","maximum");[n,r]=ss(n,r),"bool"===n.dtype&&(n=Yo(n,"int32"),r=Yo(r,"int32")),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(Vt,a)}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vl=Is({mean_:function(e,t=null,n=!1){const r={x:ks(e,"x","mean")},a={axis:t,keepDims:n};return fs.runKernel(Kt,r,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ul(e,t="float32"){if(q(e),"complex64"===t){const t=Ul(e,"float32"),n=Ul(e,"float32");return As(t,n)}const n=j(f(e),t);return fs.makeTensor(n,e,t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jl(e,t="float32"){if(q(e),"complex64"===t){const t=jl(e,"float32"),n=Ul(e,"float32");return As(t,n)}const n=U(f(e),t);return fs.makeTensor(n,e,t)}
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wl(e,t,{indexing:n="xy"}={}){if("xy"!==n&&"ij"!==n)throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(void 0===e)return[];let r=ks(e,"x","meshgrid",e instanceof Va?e.dtype:"float32");if(void 0===t)return[r];let a=ks(t,"y","meshgrid",t instanceof Va?t.dtype:"float32");const s=f(r.shape),o=f(a.shape);return"xy"===n?(r=Di(r,[1,-1]),a=Di(a,[-1,1]),[Bi(jl([o,1],r.dtype),r),Bi(a,jl([1,s],a.dtype))]):(r=Di(r,[-1,1]),a=Di(a,[1,-1]),[Bi(r,jl([1,o],r.dtype)),Bi(jl([s,1],a.dtype),a)])}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ql=Is({minimum_:function(e,t){let n=ks(e,"a","minimum"),r=ks(t,"b","minimum");[n,r]=ss(n,r),"bool"===n.dtype&&(n=Yo(n,"int32"),r=Yo(r,"int32")),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(Jt,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gl=Is({mirrorPad_:function(e,t,n){d("reflect"===n||"symmetric"===n,()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const r=ks(e,"x","mirrorPad");if(0===r.rank)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");d(t.length===r.rank,()=>`Padding doesn't match input. Must be ${r.rank}. Got ${t.length}.`);const a="reflect"===n?1:0;for(let i=0;i<r.rank;i++)d(2===t[i].length,()=>"Invalid number of paddings. Must be length of 2 each."),d(t[i][0]>=0&&t[i][0]<=r.shape[i]-a&&t[i][1]>=0&&t[i][1]<=r.shape[i]-a,()=>`Padding in dimension ${i} cannot be greater than or equal to ${r.shape[i]-a} or less than 0 for input of shape ${r.shape}`);const s={paddings:t,mode:n},o={x:r};return fs.runKernel(Zt,o,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kl=Is({mod_:function(e,t){let n=ks(e,"a","mod"),r=ks(t,"b","mod");[n,r]=ss(n,r);const a={a:n,b:r};return fs.runKernel("Mod",a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hl=Is({moments_:function(e,t=null,n=!1){const r=T(t,(e=ks(e,"x","moments")).shape),a=Vl(e,r,n);let s=a.shape;n||(s=Fu(a.shape,r));const o=Gu(_l(Yo(e,"float32"),Di(a,s)));return{mean:a,variance:Vl(o,r,n)}}});const Jl=Is({multiRNNCell_:function(e,t,n,r){const a=ks(t,"data","multiRNNCell"),s=Es(n,"c","multiRNNCell"),o=Es(r,"h","multiRNNCell");let i=a;const u=[];for(let p=0;p<e.length;p++){const t=e[p](i,s[p],o[p]);u.push(t[0]),u.push(t[1]),i=t[1]}const l=[],c=[];for(let p=0;p<u.length;p+=2)l.push(u[p]),c.push(u[p+1]);return[l,c]}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zl=Is({multinomial_:function(e,t,n,r=!1){const a=ks(e,"logits","multinomial"),s=a.size,o=a.rank;if(s<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${s}.`);if(o>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n=n||Math.random();const i={logits:1===o?Di(a,[1,-1]):a},u={numSamples:t,seed:n,normalized:r},l=fs.runKernel(Yt,i,u);return 1===o?Di(l,[l.size]):l}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xl=Is({notEqual_:function(e,t){let n=ks(e,"a","notEqual","string_or_numeric"),r=ks(t,"b","notEqual","string_or_numeric");[n,r]=ss(n,r),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(tn,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yl=Is({oneHot_:function(e,t,n=1,r=0,a="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const s={indices:ks(e,"indices","oneHot","int32")},o={dtype:a,depth:t,onValue:n,offValue:r};return fs.runKernel(on,s,o)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ql=Is({onesLike_:function(e){const t={x:ks(e,"x","onesLike")};return fs.runKernel(sn,t)}});const ec=Is({outerProduct_:function(e,t){const n=ks(e,"v1","outerProduct"),r=ks(t,"v2","outerProduct");d(1===n.rank&&1===r.rank,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${r.rank}.`);const a=Di(n,[-1,1]),s=Di(r,[1,-1]);return Bi(a,s)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tc=Is({pad_:function(e,t,n=0){const r=ks(e,"x","pad");if(0===r.rank)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const a={paddings:t,constantValue:n},s={x:r};return fs.runKernel(ln,s,a)}});const nc=Is({pad1d_:function(e,t,n=0){return d(2===t.length,()=>"Invalid number of paddings. Must be length of 2."),tc(e,[t],n)}});const rc=Is({pad2d_:function(e,t,n=0){return d(2===t.length&&2===t[0].length&&2===t[1].length,()=>"Invalid number of paddings. Must be length of 2 each."),tc(e,t,n)}});const ac=Is({pad3d_:function(e,t,n=0){return d(3===t.length&&2===t[0].length&&2===t[1].length&&2===t[2].length,()=>"Invalid number of paddings. Must be length of 2 each."),tc(e,t,n)}});const sc=Is({pad4d_:function(e,t,n=0){return d(4===t.length&&2===t[0].length&&2===t[1].length&&2===t[2].length&&2===t[3].length,()=>"Invalid number of paddings. Must be length of 2 each."),tc(e,t,n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oc=Is({spaceToBatchND_:function(e,t,n){const r=ks(e,"x","spaceToBatchND");d(r.rank>=1+t.length,()=>`input rank ${r.rank} should be > than [blockShape] ${t.length}`),d(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),d(r.shape.reduce((e,r,a)=>a>0&&a<=t.length?e&&(r+n[a-1][0]+n[a-1][1])%t[a-1]===0:e,!0),()=>`input spatial dimensions ${r.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const a={x:r},s={blockShape:t,paddings:n};return fs.runKernel(jn,a,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ic=Is({pool_:function(e,t,n,r,a,s,o){null==a&&(a=[1,1]),null==s&&(s=1),0===r&&(r="valid");const i=ks(e,"x","maxPool");let u=i,l=!1;3===i.rank&&(l=!0,u=Di(i,[1,i.shape[0],i.shape[1],i.shape[2]])),d(Ai(s,a),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${s} and dilations '${a}'`);const c=wi(u.shape,t,s,a,r),p=[c.dilationHeight,c.dilationWidth];let h;h="same"===r?function(e,t){const n=e.map((e,n)=>e+(e-1)*(t[n]-1)),r=n.map(e=>e-1),a=r.map(e=>Math.floor(e/2)),s=r.map((e,t)=>e-a[t]);return r.map((e,t)=>[a[t],s[t]])}([c.filterHeight,c.filterWidth],p):[[0,0],[0,0]];const m=1===p[0]&&1===p[1],[f,g]=function(e,t,n){const r=n.map(e=>e[0]),a=n.map(e=>e[1]),s=e.concat(r,a),o=t.map((e,t)=>(e-s[t]%e)%e),i=a.map((e,t)=>e+o[t]),u=t.map((e,t)=>[r[t],i[t]]),l=t.map((e,t)=>[0,o[t]]);return[u,l]}([c.inHeight,c.inWidth],p,h),y=m?r:"valid",b=m?u:oc(u,p,f),w=("avg"===n?()=>Ci(b,t,s,y,o):()=>Bl(b,t,s,y,o))(),N=m?w:Ui(w,p,g);return l?Di(N,[N.shape[1],N.shape[2],N.shape[3]]):N}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uc=Is({prelu_:function(e,t){const n={x:ks(e,"x","prelu"),alpha:ks(t,"alpha","prelu")};return fs.runKernel(dn,n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lc=Is({prod_:function(e,t=null,n=!1){let r=ks(e,"x","prod");"bool"===r.dtype&&(r=Yo(r,"int32"));const a={x:r},s={axis:t,keepDims:n};return fs.runKernel(hn,a,s)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cc=Is({raggedGather_:function(e,t,n,r){const a={paramsNestedSplits:e.map((e,t)=>ks(e,`tensors${t}`,"raggedGather","int32")),paramsDenseValues:ks(t,"paramsDenseValues","raggedGather"),indices:ks(n,"indices","raggedGather","int32")},s={outputRaggedRank:r},o=fs.runKernel(mn,a,s);return{outputNestedSplits:o.slice(0,o.length-1),outputDenseValues:o[o.length-1]}}});
/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pc=Is({raggedRange_:function(e,t,n){const r=ks(e,"starts","raggedRange"),a={starts:r,limits:ks(t,"limits","raggedRange",r.dtype),deltas:ks(n,"deltas","raggedRange",r.dtype)},s=fs.runKernel(fn,a);return{rtNestedSplits:s[0],rtDenseValues:s[1]}}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dc=Is({raggedTensorToTensor_:function(e,t,n,r,a){const s=ks(e,"shape","raggedTensorToTensor","int32"),o=ks(t,"values","raggedTensorToTensor"),i={shape:s,values:o,defaultValue:ks(n,"defaultValue","raggedTensorToTensor",o.dtype),rowPartitionTensors:r.map((e,t)=>ks(e,`tensors${t}`,"raggedTensorToTensor","int32"))},u={rowPartitionTypes:a};return fs.runKernel(gn,i,u)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hc=Is({rand_:function(e,t,n){q(e);const r=f(e);let a=null;if(null==n||"float32"===n)a=new Float32Array(r);else if("int32"===n)a=new Int32Array(r);else{if("bool"!==n)throw new Error(`Unknown data type ${n}`);a=new Uint8Array(r)}for(let s=0;s<r;s++)a[s]=t();return fs.makeTensor(a,e,n)}});var mc={exports:{}};!function(e,t){function n(e){var t=this,n=function(){var e=4022871197,t=function(t){t=String(t);for(var n=0;n<t.length;n++){var r=.02519603282416938*(e+=t.charCodeAt(n));r-=e=r>>>0,e=(r*=e)>>>0,e+=4294967296*(r-=e)}return 2.3283064365386963e-10*(e>>>0)};return t}();t.next=function(){var e=2091639*t.s0+2.3283064365386963e-10*t.c;return t.s0=t.s1,t.s1=t.s2,t.s2=e-(t.c=0|e)},t.c=1,t.s0=n(" "),t.s1=n(" "),t.s2=n(" "),t.s0-=n(e),t.s0<0&&(t.s0+=1),t.s1-=n(e),t.s1<0&&(t.s1+=1),t.s2-=n(e),t.s2<0&&(t.s2+=1),n=null}function r(e,t){return t.c=e.c,t.s0=e.s0,t.s1=e.s1,t.s2=e.s2,t}function a(e,t){var a=new n(e),s=t&&t.state,o=a.next;return o.int32=function(){return 4294967296*a.next()|0},o.double=function(){return o()+11102230246251565e-32*(2097152*o()|0)},o.quick=o,s&&("object"==typeof s&&r(s,a),o.state=function(){return r(a,{})}),o}t&&t.exports?t.exports=a:this.alea=a}(0,mc);var fc=mc.exports,gc={exports:{}};!function(e,t){function n(e){var t=this,n="";t.x=0,t.y=0,t.z=0,t.w=0,t.next=function(){var e=t.x^t.x<<11;return t.x=t.y,t.y=t.z,t.z=t.w,t.w^=t.w>>>19^e^e>>>8},e===(0|e)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=0|n.charCodeAt(r),t.next()}function r(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t}function a(e,t){var a=new n(e),s=t&&t.state,o=function(){return(a.next()>>>0)/4294967296};return o.double=function(){do{var e=((a.next()>>>11)+(a.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=a.next,o.quick=o,s&&("object"==typeof s&&r(s,a),o.state=function(){return r(a,{})}),o}t&&t.exports?t.exports=a:this.xor128=a}(0,gc);var yc=gc.exports,bc={exports:{}};!function(e,t){function n(e){var t=this,n="";t.next=function(){var e=t.x^t.x>>>2;return t.x=t.y,t.y=t.z,t.z=t.w,t.w=t.v,(t.d=t.d+362437|0)+(t.v=t.v^t.v<<4^e^e<<1)|0},t.x=0,t.y=0,t.z=0,t.w=0,t.v=0,e===(0|e)?t.x=e:n+=e;for(var r=0;r<n.length+64;r++)t.x^=0|n.charCodeAt(r),r==n.length&&(t.d=t.x<<10^t.x>>>4),t.next()}function r(e,t){return t.x=e.x,t.y=e.y,t.z=e.z,t.w=e.w,t.v=e.v,t.d=e.d,t}function a(e,t){var a=new n(e),s=t&&t.state,o=function(){return(a.next()>>>0)/4294967296};return o.double=function(){do{var e=((a.next()>>>11)+(a.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=a.next,o.quick=o,s&&("object"==typeof s&&r(s,a),o.state=function(){return r(a,{})}),o}t&&t.exports?t.exports=a:this.xorwow=a}(0,bc);var wc=bc.exports,Nc={exports:{}};!function(e,t){function n(e){var t=this;t.next=function(){var e,n,r=t.x,a=t.i;return e=r[a],n=(e^=e>>>7)^e<<24,n^=(e=r[a+1&7])^e>>>10,n^=(e=r[a+3&7])^e>>>3,n^=(e=r[a+4&7])^e<<7,e=r[a+7&7],n^=(e^=e<<13)^e<<9,r[a]=n,t.i=a+1&7,n},function(e,t){var n,r=[];if(t===(0|t))r[0]=t;else for(t=""+t,n=0;n<t.length;++n)r[7&n]=r[7&n]<<15^t.charCodeAt(n)+r[n+1&7]<<13;for(;r.length<8;)r.push(0);for(n=0;n<8&&0===r[n];++n);for(8==n?r[7]=-1:r[n],e.x=r,e.i=0,n=256;n>0;--n)e.next()}(t,e)}function r(e,t){return t.x=e.x.slice(),t.i=e.i,t}function a(e,t){null==e&&(e=+new Date);var a=new n(e),s=t&&t.state,o=function(){return(a.next()>>>0)/4294967296};return o.double=function(){do{var e=((a.next()>>>11)+(a.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=a.next,o.quick=o,s&&(s.x&&r(s,a),o.state=function(){return r(a,{})}),o}t&&t.exports?t.exports=a:this.xorshift7=a}(0,Nc);var xc=Nc.exports,vc={exports:{}};!function(e,t){function n(e){var t=this;t.next=function(){var e,n,r=t.w,a=t.X,s=t.i;return t.w=r=r+1640531527|0,n=a[s+34&127],e=a[s=s+1&127],n^=n<<13,e^=e<<17,n^=n>>>15,e^=e>>>12,n=a[s]=n^e,t.i=s,n+(r^r>>>16)|0},function(e,t){var n,r,a,s,o,i=[],u=128;for(t===(0|t)?(r=t,t=null):(t+="\0",r=0,u=Math.max(u,t.length)),a=0,s=-32;s<u;++s)t&&(r^=t.charCodeAt((s+32)%t.length)),0===s&&(o=r),r^=r<<10,r^=r>>>15,r^=r<<4,r^=r>>>13,s>=0&&(o=o+1640531527|0,a=0==(n=i[127&s]^=r+o)?a+1:0);for(a>=128&&(i[127&(t&&t.length||0)]=-1),a=127,s=512;s>0;--s)r=i[a+34&127],n=i[a=a+1&127],r^=r<<13,n^=n<<17,r^=r>>>15,n^=n>>>12,i[a]=r^n;e.w=o,e.X=i,e.i=a}(t,e)}function r(e,t){return t.i=e.i,t.w=e.w,t.X=e.X.slice(),t}function a(e,t){null==e&&(e=+new Date);var a=new n(e),s=t&&t.state,o=function(){return(a.next()>>>0)/4294967296};return o.double=function(){do{var e=((a.next()>>>11)+(a.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=a.next,o.quick=o,s&&(s.X&&r(s,a),o.state=function(){return r(a,{})}),o}t&&t.exports?t.exports=a:this.xor4096=a}(0,vc);var Tc=vc.exports,Sc={exports:{}};!function(e,t){function n(e){var t=this,n="";t.next=function(){var e=t.b,n=t.c,r=t.d,a=t.a;return e=e<<25^e>>>7^n,n=n-r|0,r=r<<24^r>>>8^a,a=a-e|0,t.b=e=e<<20^e>>>12^n,t.c=n=n-r|0,t.d=r<<16^n>>>16^a,t.a=a-e|0},t.a=0,t.b=0,t.c=-1640531527,t.d=1367130551,e===Math.floor(e)?(t.a=e/4294967296|0,t.b=0|e):n+=e;for(var r=0;r<n.length+20;r++)t.b^=0|n.charCodeAt(r),t.next()}function r(e,t){return t.a=e.a,t.b=e.b,t.c=e.c,t.d=e.d,t}function a(e,t){var a=new n(e),s=t&&t.state,o=function(){return(a.next()>>>0)/4294967296};return o.double=function(){do{var e=((a.next()>>>11)+(a.next()>>>0)/4294967296)/(1<<21)}while(0===e);return e},o.int32=a.next,o.quick=o,s&&("object"==typeof s&&r(s,a),o.state=function(){return r(a,{})}),o}t&&t.exports?t.exports=a:this.tychei=a}(0,Sc);var kc=Sc.exports,Ec={exports:{}};const _c=n(Object.freeze(Object.defineProperty({__proto__:null,default:{}},Symbol.toStringTag,{value:"Module"})));var Ic;Ic=Ec,function(e,t,n){var r,a=256,s="random",o=n.pow(a,6),i=n.pow(2,52),u=2*i,l=255;function c(l,c,g){var y=[],b=m(h((c=1==c?{entropy:!0}:c||{}).entropy?[l,f(t)]:null==l?function(){try{var n;return r&&(n=r.randomBytes)?n=n(a):(n=new Uint8Array(a),(e.crypto||e.msCrypto).getRandomValues(n)),f(n)}catch(vf){var s=e.navigator,o=s&&s.plugins;return[+new Date,e,o,e.screen,f(t)]}}():l,3),y),w=new p(y),N=function(){for(var e=w.g(6),t=o,n=0;e<i;)e=(e+n)*a,t*=a,n=w.g(1);for(;e>=u;)e/=2,t/=2,n>>>=1;return(e+n)/t};return N.int32=function(){return 0|w.g(4)},N.quick=function(){return w.g(4)/4294967296},N.double=N,m(f(w.S),t),(c.pass||g||function(e,t,r,a){return a&&(a.S&&d(a,w),e.state=function(){return d(w,{})}),r?(n[s]=e,t):e})(N,b,"global"in c?c.global:this==n,c.state)}function p(e){var t,n=e.length,r=this,s=0,o=r.i=r.j=0,i=r.S=[];for(n||(e=[n++]);s<a;)i[s]=s++;for(s=0;s<a;s++)i[s]=i[o=l&o+e[s%n]+(t=i[s])],i[o]=t;(r.g=function(e){for(var t,n=0,s=r.i,o=r.j,i=r.S;e--;)t=i[s=l&s+1],n=n*a+i[l&(i[s]=i[o=l&o+t])+(i[o]=t)];return r.i=s,r.j=o,n})(a)}function d(e,t){return t.i=e.i,t.j=e.j,t.S=e.S.slice(),t}function h(e,t){var n,r=[],a=typeof e;if(t&&"object"==a)for(n in e)try{r.push(h(e[n],t-1))}catch(vf){}return r.length?r:"string"==a?e:e+"\0"}function m(e,t){for(var n,r=e+"",a=0;a<r.length;)t[l&a]=l&(n^=19*t[l&a])+r.charCodeAt(a++);return f(t)}function f(e){return String.fromCharCode.apply(0,e)}if(m(n.random(),t),Ic.exports){Ic.exports=c;try{r=_c}catch(g){}}else n["seed"+s]=c}("undefined"!=typeof self?self:t,[],Math);var Ac=fc,Mc=yc,$c=wc,Oc=xc,Dc=Tc,Cc=kc,Rc=Ec.exports;Rc.alea=Ac,Rc.xor128=Mc,Rc.xorwow=$c,Rc.xorshift7=Oc,Rc.xor4096=Dc,Rc.tychei=Cc;var Fc=Rc;
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bc{constructor(e,t,n,r,a){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+2*this.stdDev,this.lower=this.mean-2*this.stdDev);const s=a||Math.random();this.random=Fc.alea(s.toString())}nextValue(){if(!isNaN(this.nextVal)){const e=this.nextVal;return this.nextVal=NaN,e}let e,t,n=!1;for(;!n;){let r,a,s;do{r=2*this.random()-1,a=2*this.random()-1,s=r*r+a*a}while(s>=1||0===s);const o=Math.sqrt(-2*Math.log(s)/s);e=this.mean+this.stdDev*r*o,t=this.mean+this.stdDev*a*o,this.truncated&&!this.isValidTruncated(e)||(n=!0)}return this.truncated&&!this.isValidTruncated(t)||(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return null==this.dtype||"float32"===this.dtype?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}}class zc{constructor(e,t,n,r){this.alpha=e,this.beta=1/t,this.dtype=n;const a=r||Math.random();this.randu=Fc.alea(a.toString()),this.randn=new Bc(0,1,n,!1,this.randu()),this.d=e<1?e+2/3:e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,t,n,r,a,s;for(;;){do{r=this.randn.nextValue(),s=1+this.c*r}while(s<=0);if(s*=s*s,e=r*r,t=1-.331*e*e,n=.5*e+this.d*(1-s+Math.log(s)),a=this.randu(),a<t||Math.log(a)<n)break}return s=1/this.beta*this.d*s,this.alpha<1&&(s*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(s)}convertValue(e){return"float32"===this.dtype?e:Math.round(e)}}class Lc{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>null==this.dtype||"float32"===this.dtype,this.min=e,this.range=t-e,this.dtype=n,null==r&&(r=Math.random()),"number"==typeof r&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=Fc.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pc=Is({randomGamma_:function(e,t,n=1,r="float32",a){if(q(e),null==n&&(n=1),null==r&&(r="float32"),"float32"!==r&&"int32"!==r)throw new Error(`Unsupported data type ${r}`);const s=new zc(t,n,r,a),o=Xo(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=s.nextValue();return o.toTensor()}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vc=Is({randomNormal_:function(e,t=0,n=1,r,a){if(q(e),null!=r&&"bool"===r)throw new Error(`Unsupported data type ${r}`);const s=new Bc(t,n,r,!1,a),o=Xo(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=s.nextValue();return o.toTensor()}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uc=Is({randomStandardNormal_:function(e,t,n){if(null!=t&&"bool"===t)throw new Error(`Unsupported data type ${t}`);return Vc(e,0,1,t,n)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jc=Is({randomUniform_:function(e,t=0,n=1,r="float32",a){q(e);const s=Xo(e,r),o=new Lc(t,n,null,a);for(let i=0;i<s.values.length;i++)s.values[i]=o.nextValue();return s.toTensor()}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wc=Is({randomUniformInt_:function(e,t,n,r){return jc(e,t,n,"int32",r)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qc(e,t,n=1,r="float32"){if(0===n)throw new Error("Cannot have a step of zero");const a={start:e,stop:t,step:n,dtype:r};return fs.runKernel(yn,{},a)}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gc=Is({real_:function(e){const t={input:ks(e,"input","real")};return fs.runKernel(bn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kc=Is({reciprocal_:function(e){const t={x:ks(e,"x","reciprocal")};return fs.runKernel(wn,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hc=Is({relu_:function(e){const t={x:ks(e,"x","relu")};return fs.runKernel(Nn,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jc=Is({relu6_:function(e){const t={x:ks(e,"x","relu6")};return fs.runKernel(En,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zc=Is({reverse_:function(e,t){const n={x:ks(e,"x","reverse")},r={dims:t};return fs.runKernel(_n,n,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xc=Is({reverse1d_:function(e){const t=ks(e,"x","reverse");return d(1===t.rank,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),Zc(t,0)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yc=Is({reverse2d_:function(e,t){const n=ks(e,"x","reverse");return d(2===n.rank,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),Zc(n,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qc=Is({reverse3d_:function(e,t){const n=ks(e,"x","reverse");return d(3===n.rank,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),Zc(n,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ep=Is({reverse4d_:function(e,t){const n=ks(e,"x","reverse");return d(4===n.rank,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),Zc(n,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tp=Is({round_:function(e){const t={x:ks(e,"x","round")};return fs.runKernel(In,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const np=Is({rsqrt_:function(e){const t={x:ks(e,"x","rsqrt","float32")};return fs.runKernel(An,t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rp=Is({selu_:function(e){const t={x:ks(e,"x","selu")};return fs.runKernel(Cn,t)}});const ap=Is({separableConv2d_:function(e,t,n,r,a,s=[1,1],o="NHWC"){const i=ks(e,"x","separableConv2d"),u=ks(t,"depthwiseFilter","separableConv2d"),l=ks(n,"pointwiseFilter","separableConv2d");let c=i,p=!1;if(3===i.rank&&(p=!0,c=Di(i,[1,i.shape[0],i.shape[1],i.shape[2]])),"NCHW"===o)throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");d(4===c.rank,()=>`Error in separableConv2d: input must be rank 4, but got rank ${c.rank}.`),d(4===u.rank,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${u.rank}.`),d(4===l.rank,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${u.rank}.`),d(1===l.shape[0],()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),d(1===l.shape[1],()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);const h=u.shape[2],m=u.shape[3];d(l.shape[2]===h*m,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${h*m}, but got ${l.shape[2]}.`);const f=yu(c,u,r,a,o,s),g=au(f,l,1,"valid",o);return p?Di(g,[g.shape[1],g.shape[2],g.shape[3]]):g}});
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sp=async function(e,t){const n=ks(e,"x","setdiff1d"),r=ks(t,"y","setdiff1d");d(n.dtype===r.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${r.dtype}).`),d(1===n.rank,()=>`x should be 1D tensor, but got x (${n.shape}).`),d(1===r.rank,()=>`y should be 1D tensor, but got y (${r.shape}).`);const a=await n.data(),s=await r.data(),o=new Set(s);let i=0;for(let c=0;c<a.length;c++)o.has(a[c])||i++;const u=new za([i],n.dtype),l=new za([i],"int32");for(let c=0,p=0;c<a.length;c++)o.has(a[c])||(u.values[p]=a[c],l.values[p]=c,p++);return[u.toTensor(),l.toTensor()]};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const op=Is({sign_:function(e){const t={x:ks(e,"x","sign")};return fs.runKernel(zn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ip=Is({sin_:function(e){const t={x:ks(e,"x","sin","float32")};return fs.runKernel("Sin",t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const up=Is({sinh_:function(e){const t={x:ks(e,"x","sinh")};return fs.runKernel(Bn,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lp=Is({slice1d_:function(e,t,n){const r=ks(e,"x","slice1d");return d(1===r.rank,()=>`slice1d expects a rank-1 tensor, but got a rank-${r.rank} tensor`),Li(r,[t],[n])}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cp=Is({slice2d_:function(e,t,n){const r=ks(e,"x","slice2d");return d(2===r.rank,()=>`slice2d expects a rank-2 tensor, but got a rank-${r.rank} tensor`),Li(r,t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pp=Is({slice3d_:function(e,t,n){const r=ks(e,"x","slice3d");return d(3===r.rank,()=>`slice3d expects a rank-3 tensor, but got a rank-${r.rank} tensor`),Li(r,t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dp=Is({slice4d_:function(e,t,n){const r=ks(e,"x","slice4d");return d(4===r.rank,()=>`slice4d expects a rank-4 tensor, but got a rank-${r.rank} tensor`),Li(r,t,n)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hp=Is({softmax_:function(e,t=-1){const n=ks(e,"logits","softmax","float32");if(-1===t&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const r={logits:n},a={dim:t};return fs.runKernel(qn,r,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mp=Is({fft_:function(e){d("complex64"===e.dtype,()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);const t={input:e};return fs.runKernel("FFT",t)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fp=Is({ifft_:function(e){d("complex64"===e.dtype,()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);const t={input:e};return fs.runKernel(Nt,t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gp=Is({irfft_:function(e){const t=e.shape[e.shape.length-1],n=e.size/t;let r;if(t<=2){const a=Di(e,[n,t]);r=fp(a)}else{const a=[n,2*(t-1)],s=Di(Gc(e),[n,t]),o=Di(ol(e),[n,t]),i=Zc(Li(s,[0,1],[n,t-2]),1),u=ai(Zc(Li(o,[0,1],[n,t-2]),1),Wu(-1)),l=Fi([s,i],1),c=Fi([o,u],1),p=Di(As(l,c),[a[0],a[1]]);r=fp(p)}if(r=Gc(r),3===e.rank&&0!==e.shape[0]){const t=r,n=e.shape[0];r=Di(r,[n,r.shape[0]/n,r.shape[1]]),t.dispose()}return r}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yp=Is({split_:function(e,t,n=0){const r={x:ks(e,"x","split")},a={numOrSizeSplits:t,axis:n};return fs.runKernel(Wn,r,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bp=Is({rfft_:function(e,t){d("float32"===e.dtype,()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1];const r=e.size/n;let a;if(null!=t&&t<n){const r=e.shape.map(e=>0),s=e.shape.map(e=>e);s[e.shape.length-1]=t,a=Li(e,r,s),n=t}else if(null!=t&&t>n){const r=e.shape.map(e=>e);r[e.shape.length-1]=t-n,a=Fi([e,Ul(r)],e.shape.length-1),n=t}else a=e;const s=Eu(a),o=Di(As(a,s),[r,n]),i=mp(o),u=Math.floor(n/2)+1,l=Gc(i),c=ol(i),p=yp(l,[u,n-u],l.shape.length-1),h=yp(c,[u,n-u],c.shape.length-1),m=a.shape.slice();return m[a.shape.length-1]=u,Di(As(p[0],h[0]),m)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wp=Is({squaredDifference_:function(e,t){let n=ks(e,"a","squaredDifference"),r=ks(t,"b","squaredDifference");[n,r]=ss(n,r),vu(n.shape,r.shape);const a={a:n,b:r};return fs.runKernel(Xn,a,{})}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Np=Is({squeeze_:function(e,t){const n=ks(e,"x","squeeze","string_or_numeric");return Di(n,S(n.shape,t).newShape)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xp=Is({stack_:function(e,t=0){const n=Es(e,"tensors","stack","string_or_numeric");d(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&d(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const r=n,a={axis:t};return fs.runKernel(un,r,a)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vp=Is({step_:function(e,t=0){const n={x:ks(e,"x","step")},r={alpha:t};return fs.runKernel(gr,n,r)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tp=Is({stridedSlice_:function(e,t,n,r,a=0,s=0,o=0,i=0,u=0){const l={x:ks(e,"x","stridedSlice","string_or_numeric")},c={begin:t,end:n,strides:r,beginMask:a,endMask:s,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:u};return fs.runKernel(er,l,c)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sp=Is({tan_:function(e){const t={x:ks(e,"x","tan","float32")};return fs.runKernel("Tan",t)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kp(e,t){m(e);const n=vs(e,t);if(1!==n.length)throw new Error("tensor1d() requires values to be a flat/TypedArray");return Ms(e,null,n,t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ep(e,t,n){if(m(e),null!=t&&2!==t.length)throw new Error("tensor2d() requires shape to have two numbers");const r=vs(e,n);if(2!==r.length&&1!==r.length)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return Ms(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _p(e,t,n){if(m(e),null!=t&&3!==t.length)throw new Error("tensor3d() requires shape to have three numbers");const r=vs(e,n);if(3!==r.length&&1!==r.length)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return Ms(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ip(e,t,n){if(m(e),null!=t&&4!==t.length)throw new Error("tensor4d() requires shape to have four numbers");const r=vs(e,n);if(4!==r.length&&1!==r.length)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return Ms(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ap(e,t,n){if(m(e),null!=t&&5!==t.length)throw new Error("tensor5d() requires shape to have five numbers");const r=vs(e,n);if(5!==r.length&&1!==r.length)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return Ms(e,t,r,n)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mp(e,t,n){if(m(e),null!=t&&6!==t.length)throw new Error("tensor6d() requires shape to have six numbers");const r=vs(e,n);if(6!==r.length&&1!==r.length)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(1===r.length&&null==t)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return Ms(e,t=t||r,r,n)}function $p(e,t,n){const r=t.rank>1?t.shape[t.rank-1]:1,a=t.rank>1?t.rank-1:1,s=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${a}.`;if(n.rank<a)throw new Error(s+` update.rank < ${a}. `);if(e.length<r+(n.rank-a))throw new Error(s+` Output shape length < ${r+(n.rank-a)}`);if(n.rank!==a+e.length-r)throw new Error(s+" update.rank != "+(a+e.length-r));for(let o=0;o<a;++o)if(n.shape[o]!==t.shape[o])throw new Error(s+` updates.shape[${o}] (${n.shape[o]}) != indices.shape[${o}] (${t.shape[o]}).`);for(let o=0;o<n.rank-a;++o)if(n.shape[o+a]!==e[o+r])throw new Error(s+` updates.shape[${o+a}] (${n.shape[o+a]}) != shape[${o+a}] (${e[o+a]})`)}function Op(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if("int32"!==t.dtype)throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(0===n.length){if(0===t.size)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(0===e.size)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}$p(n,t,e)}function Dp(e,t,n){const r=t.shape.length,a=r>1?t.shape[r-1]:1,s=n.length;let o=1;for(let u=a;u<s;++u)o*=n[u];const i=a<1?1:a;return{sliceRank:a,numUpdates:f(t.shape)/i,sliceSize:o,strides:[...z(n.slice(0,a)),1],outputSize:f(n)}}const Cp=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:Dp,validateInput:Op,validateUpdateShape:$p},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rp=Is({tensorScatterUpdate_:function(e,t,n){const r=ks(e,"tensor","tensorScatterupdate"),a=ks(t,"indices","tensorScatterupdate","int32"),s=ks(n,"updates","tensorScatterupdate");if(Op(s,a,r.shape),r.dtype!==s.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${r.dtype} and ${s.dtype}.`);const o={tensor:r,indices:a,updates:s};return fs.runKernel($n,o,{})}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fp=Is({topk_:function(e,t=1,n=!0){const r=ks(e,"x","topk");if(0===r.rank)throw new Error("topk() expects the input to be of rank 1 or higher");const a=r.shape[r.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>a)throw new Error(`'k' passed to topk() must be <= the last dimension (${a}) but got ${t}`);const s={x:r},o={k:t,sorted:n},[i,u]=fs.runKernel(ur,s,o);return{values:i,indices:u}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bp=Is({truncatedNormal_:function(e,t=0,n=1,r,a){if(q(e),null!=r&&"bool"===r)throw new Error("Unsupported data type $ { dtype }");const s=new Bc(t,n,r,!0,a),o=Xo(e,r);for(let i=0;i<o.values.length;i++)o.values[i]=s.nextValue();return o.toTensor()}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zp=Is({unique_:function(e,t=0){const n=ks(e,"x","unique","string_or_numeric");d(n.rank>0,()=>"The input tensor must be at least 1D");const r={x:n},a={axis:t},[s,o]=fs.runKernel(pr,r,a);return{values:s,indices:o}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Lp=Is({unsortedSegmentSum_:function(e,t,n){const r=ks(e,"x","unsortedSegmentSum"),a=ks(t,"segmentIds","unsortedSegmentSum","int32");d(b(n),()=>"numSegments must be of dtype int");const s={x:r,segmentIds:a},o={numSegments:n};return fs.runKernel(hr,s,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pp=Is({unstack_:function(e,t=0){const n=ks(e,"x","unstack","string_or_numeric");d(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const r={value:n},a={axis:t};return fs.runKernel(dr,r,a)}});
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vp(e,t){return Rl(e,t,"right")}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Up(e,t=!0,n,r){return fs.makeVariable(e,t,n,r)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jp(e,t){const n=[];for(let s=0;s<t.length;s++)t[s]&&n.push(s);const r=Xo(e,"int32"),a=Xo([n.length,e.length],"int32");for(let s=0;s<n.length;s++){const t=r.indexToLoc(n[s]),o=s*e.length;a.values.set(t,o)}return a.toTensor()}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wp=async function(e){const t=ks(e,"condition","whereAsync","bool"),n=await t.data(),r=jp(t.shape,n);return e!==t&&t.dispose(),r};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qp=async function(e,t,n){const r=ks(e,"tensor","boolMask"),a=ks(t,"mask","boolMask","bool"),s=null==n?0:n,o=a.rank,i=r.shape;d(o>0,()=>"mask cannot be scalar"),h(i.slice(s,s+o),a.shape,"mask's shape must match the first K dimensions of tensor's shape,");let u=1;for(let d=s;d<s+o;d++)u*=i[d];const l=i.slice(0,s).concat([u],i.slice(s+o)),c=Di(r,l),p=Di(a,[-1]),m=await Wp(p),f=Np(m,[1]),g=rl(c,f,s);return e!==r&&r.dispose(),t!==a&&a.dispose(),f.dispose(),c.dispose(),p.dispose(),m.dispose(),g};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gp=Is({transpose_:function(e,t,n){const r=ks(e,"x","transpose");if(null==t&&(t=r.shape.map((e,t)=>t).reverse()),d(r.rank===t.length,()=>`Error in transpose: rank of input ${r.rank} must match length of perm ${t}.`),t.forEach(e=>{d(e>=0&&e<r.rank,()=>`All entries in 'perm' must be between 0 and ${r.rank-1} but got ${t}`)}),r.rank<=1)return r.clone();const a={x:r},s={perm:t};return"complex64"===r.dtype?Us(()=>{let e=Gc(r),t=ol(r);return e=fs.runKernel(cr,{x:e},s),t=fs.runKernel(cr,{x:t},s),n&&(t=Sl(t)),As(e,t)}):fs.runKernel(cr,a,s)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Kp=Is({movingAverage_:function(e,t,n,r,a=!0){const s=ks(e,"v","movingAverage"),o=ks(t,"x","movingAverage"),i=ks(n,"decay","movingAverage");os(s,o),d(y(s.shape,o.shape),()=>"Shape mismatch in v and x");const u=Wu(1),l=_l(u,i);let c=ai(_l(o,s),l);if(a){d(null!=r,()=>"When using zeroDebias: true, step is required.");const e=ks(r,"step","movingAverage");c=ri(c,_l(u,ju(i,e)))}return ti(s,c)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hp=Is({scatterND_:function(e,t,n){q(n);const r=ks(e,"indices","scatterND","int32"),a=ks(t,"updates","scatterND");Op(a,r,n);const s={indices:r,updates:a},o={shape:n};return fs.runKernel(Mn,s,o)}});const Jp=Is({sparseToDense_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r=0){q(n);const a=ks(e,"sparseIndices","sparseToDense","int32"),s=ks(t,"sparseValues","sparseToDense","string_or_numeric"),o=ks(r,"defaultValue","sparseToDense",s.dtype);!function(e,t,n,r){if("int32"!==e.dtype)throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);const a=e.rank>0?e.shape[0]:1,s=e.rank>1?e.shape[1]:1;if(n.length!==s)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${s}.`);const o=t.size;if(0!==t.rank&&(1!==t.rank||o!==a))throw new Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${a}]`);if(t.dtype!==r.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}(a,s,n,o);const i={sparseIndices:a,sparseValues:s,defaultValue:o},u={outputShape:n};return fs.runKernel(Zn,i,u)}});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zp=Is({gatherND_:function(e,t){const n=ks(t,"indices","gatherND","int32"),r={params:ks(e,"x","gatherND","string_or_numeric"),indices:n};return fs.runKernel(gt,r)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xp=Is({dropout_:
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r){const a=ks(e,"x","dropout");if(d("float32"===a.dtype,()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${a.dtype} tensor instead.`),d(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),0===t)return e instanceof Va?a.clone():a;const s=function(e,t){if(null==t)return e.shape.slice();if(y(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let r=0;r<e.shape.length;r++)null==t[r]&&null!=e.shape[r]?n.push(e.shape[r]):n.push(t[r]);return n}return t}(a,n),o=1-t,i=ri(nl(ti(jc(s,0,1,"float32",r),o)),o);return ai(a,i)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yp(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function Qp(e,t,n){const r=1-e%2,a=new Float32Array(e);for(let s=0;s<e;++s){const o=2*Math.PI*s/(e+r-1);a[s]=t-n*Math.cos(o)}return kp(a,"float32")}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ed=async function(e,t,n=1){const r=ks(e,"predictions","inTopK"),a=ks(t,"targets","inTopK");d(r.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${r.rank}`),d(r.rank-1===a.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${r.rank} and targets rank ${a.rank}`),h(r.shape.slice(0,r.shape.length-1),a.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const s=r.shape[r.shape.length-1];d(n>0&&n<=s,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${s}), but got ${n}`);const o=await r.data(),i=await a.data(),[u,l]=[o.length/s,s],c=k("bool",u);for(let p=0;p<u;p++){const e=p*l,t=o.subarray(e,e+l),r=[];for(let n=0;n<t.length;n++)r.push({value:t[n],index:n});r.sort((e,t)=>t.value-e.value),c[p]=0;for(let a=0;a<n;a++)if(r[a].index===i[p]){c[p]=1;break}}return e!==r&&r.dispose(),t!==a&&a.dispose(),$s(c,a.shape,"bool")};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const td=Is({conv2DBackpropFilter_:function(e,t,n,r,a,s="NHWC",o){let i=e;3===e.rank&&(i=Di(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;3===u.rank&&(u=Di(t,[1,t.shape[0],t.shape[1],t.shape[2]])),d(4===i.rank,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),d(4===u.rank,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${u.shape}.`),d(4===n.length,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const l="NHWC"===s?i.shape[3]:i.shape[1],c="NHWC"===s?u.shape[3]:u.shape[1];d(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),d(c===n[3],()=>`Error in conv2dDerFilter: depth of dy (${c}) must match output depth for filter (${n[3]}).`),Oi("conv2dDerFilter",a,o);const p={x:i,dy:u},h={strides:r,pad:a,dataFormat:s,dimRoundingMode:o,filterShape:n};return fs.runKernel(Ce,p,h)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nd(e,t,n){if(null==n||"linear"===n)return e;if("relu"===n)return ai(e,vp(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function rd(e,t){let n=t;const r=xu(e.shape,t.shape);return r.length>0&&(n=Ku(n,r)),Di(n,e.shape)}function ad(e,t,n,r){if("linear"===t)return e;if("relu"===t)return Hc(e);if("elu"===t)return Mu(e);if("relu6"===t)return Jc(e);if("prelu"===t)return uc(e,n);if("leakyrelu"===t)return cl(e,r);if("sigmoid"===t)return zi(e);throw new Error(`Unknown fused activation ${t}.`)}const sd=(e,t)=>!(e>0)||"linear"===t;
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const od=Is({fusedConv2d_:function({x:e,filter:t,strides:n,pad:r,dataFormat:a="NHWC",dilations:s=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:l,leakyreluAlpha:c}){if(u=u||"linear",!1===sd(fs.state.gradientDepth,u)){d("NHWC"===a,()=>`Error in fused conv2d: got dataFormat of ${a} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let p=au(e,t,n,r,a,s,o);return null!=i&&(p=ti(p,i)),ad(p,u,l,c)}const p=ks(e,"x","conv2d","float32"),h=ks(t,"filter","conv2d","float32");let m=p,f=!1;3===p.rank&&(f=!0,m=Di(p,[1,p.shape[0],p.shape[1],p.shape[2]])),d(4===m.rank,()=>`Error in fused conv2d: input must be rank 4, but got rank ${m.rank}.`),d(4===h.rank,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${h.rank}.`),Oi("fused conv2d",r,o);const g="NHWC"===a?m.shape[3]:m.shape[1];d(h.shape[2]===g,()=>`Error in conv2d: depth of input (${g}) must match input depth for filter ${h.shape[2]}.`),d(Ai(n,s),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${s}'`);const y=xi(m.shape,h.shape,n,s,r,o);let b,w;if(null!=i&&(b=ks(i,"bias","fused conv2d"),[b]=ss(b,p),"NHWC"===a?vu(y.outShape,b.shape):(d(b.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${b.shape.length}.`),d(0===b.shape.length||b.shape[0]===y.outChannels||1===b.shape[0],()=>`Error in fused conv2d: bias shape (${b.shape}) is not compatible with the number of output channels (${y.outChannels})`))),null!=l){const e=l.shape;if(d(e.length<=1||3===e.length,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${e.length}.`),1===e.length)d(1===e[0]||e[0]===y.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the number of output channels (${y.outChannels}).`);else if(3===e.length)try{vu(e,y.outShape)}catch(vf){const n=`Error in fused conv2d: PReLU activation weights (${e}) is not compatible with the output shape of the conv2d (${y.outShape}).`;throw Error(n)}w=ks(l,"prelu weights","fused conv2d")}const N=(e,t)=>{d("NHWC"===a,()=>`Error in gradient of fused conv2D: got dataFormat of ${a} but only NHWC is currently supported.`);const[o,i,l,c]=t,p=nd(e,l,u);d(Ii(s),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${s}'`);const h=[ou(i.shape,p,o,n,r),td(i,p,o.shape,n,r)];if(null!=c){const e=rd(c,p);h.push(e)}return h},x={x:m,filter:h,bias:b,preluActivationWeights:w},v={strides:n,pad:r,dataFormat:a,dilations:s,dimRoundingMode:o,activation:u,leakyreluAlpha:c};if(null==i){return vl((e,t,n)=>{let r=fs.runKernel(Nr,x,v);return n([t,e,r]),f&&(r=Di(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:N}})(m,h)}return vl((e,t,n,r)=>{let a=fs.runKernel(Nr,x,v);return r([t,e,a,n]),f&&(a=Di(a,[a.shape[1],a.shape[2],a.shape[3]])),{value:a,gradFunc:N}})(m,h,b)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const id=Is({depthwiseConv2dNativeBackpropFilter_:function(e,t,n,r,a,s=[1,1],o){let i=e;3===e.rank&&(i=Di(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;3===u.rank&&(u=Di(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const l={x:i,dy:u},c={strides:r,pad:a,dimRoundingMode:o,dilations:s,filterShape:n};return fs.runKernel(Ke,l,c)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ud=Is({depthwiseConv2dNativeBackpropInput_:function(e,t,n,r,a,s=[1,1],o){let i=t,u=!1;3===t.rank&&(u=!0,i=Di(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const l={dy:i,filter:n},c={strides:r,pad:a,dimRoundingMode:o,dilations:s,inputShape:e},p=fs.runKernel(He,l,c);return u?Di(p,[p.shape[1],p.shape[2],p.shape[3]]):p}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ld=Is({fusedDepthwiseConv2d_:function({x:e,filter:t,strides:n,pad:r,dataFormat:a="NHWC",dilations:s=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:l,leakyreluAlpha:c}){if(!1===sd(fs.state.gradientDepth,u)){let p=yu(e,t,n,r,a,s,o);return null!=i&&(p=ti(p,i)),ad(p,u,l,c)}const p=ks(e,"x","depthwiseConv2d","float32"),h=ks(t,"filter","depthwiseConv2d","float32");let m=p,f=!1;3===p.rank&&(f=!0,m=Di(p,[1,p.shape[0],p.shape[1],p.shape[2]])),d(4===m.rank,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${m.rank}.`),d(4===h.rank,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${h.rank}.`),d(m.shape[3]===h.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${m.shape[3]}) must match the inChannels dimension in filter ${h.shape[2]}.`),null==s&&(s=[1,1]),d(Ai(n,s),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${s}'`),Oi("fused depthwiseConv2d",r,o);const g=xi(m.shape,h.shape,n,s,r,o,!0);let y,b;null!=i&&(y=ks(i,"bias","fused conv2d"),[y]=ss(y,p),vu(g.outShape,y.shape)),null!=l&&(b=ks(l,"prelu weights","fused depthwiseConv2d"));const w=(e,t)=>{d(Ii(s),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${s}'`);const[a,i,l,c]=t,p=nd(e,l,u),h=ud(i.shape,p,a,n,r,s,o),m=id(i,p,a.shape,n,r,s,o);if(null!=c){return[h,m,rd(y,p)]}return[h,m]},N={x:m,filter:h,bias:y,preluActivationWeights:b},x={strides:n,pad:r,dataFormat:a,dilations:s,dimRoundingMode:o,activation:u,leakyreluAlpha:c};if(null==i){return vl((e,t,n)=>{let r=fs.runKernel(xr,N,x);return n([t,e,r]),f&&(r=Di(r,[r.shape[1],r.shape[2],r.shape[3]])),{value:r,gradFunc:w}})(m,h)}return vl((e,t,n,r)=>{let a=fs.runKernel(xr,N,x);return r([t,e,a,n]),f&&(a=Di(a,[a.shape[1],a.shape[2],a.shape[3]])),{value:a,gradFunc:w}})(m,h,y)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cd=Is({fusedMatMul_:function({a:e,b:t,transposeA:n=!1,transposeB:r=!1,bias:a,activation:s="linear",preluActivationWeights:o,leakyreluAlpha:i=.2}){if(!1===sd(fs.state.gradientDepth,s)){let u=Bi(e,t,n,r);return null!=a&&(u=ti(u,a)),ad(u,s,o,i)}let u=ks(e,"a","fused matMul"),l=ks(t,"b","fused matMul");[u,l]=ss(u,l);const c=n?u.shape[u.rank-2]:u.shape[u.rank-1],p=r?l.shape[l.rank-1]:l.shape[l.rank-2],h=n?u.shape[u.rank-1]:u.shape[u.rank-2],m=r?l.shape[l.rank-2]:l.shape[l.rank-1],g=u.shape.slice(0,-2),y=l.shape.slice(0,-2),b=f(g),w=f(y);d(c===p,()=>`Error in fused matMul: inner shapes (${c}) and (${p}) of Tensors with shapes ${u.shape} and ${l.shape} and transposeA=${n} and transposeB=${r} must match.`);const N=vu(u.shape.slice(0,-2),l.shape.slice(0,-2)).concat([h,m]),x=Di(u,n?[b,c,h]:[b,h,c]),v=Di(l,r?[w,m,p]:[w,p,m]);let T,S;null!=a&&(T=ks(a,"bias","fused matMul"),[T]=ss(T,u),vu(N,T.shape)),null!=o&&(S=ks(o,"prelu weights","fused matMul"));const k=(e,t)=>{const[o,i,u,l]=t,c=nd(Di(e,u.shape),u,s);let p,d;if(n||r?!n&&r?(p=Bi(c,i,!1,!1),d=Bi(c,o,!0,!1)):n&&!r?(p=Bi(i,c,!1,!0),d=Bi(o,c,!1,!1)):(p=Bi(i,c,!0,!0),d=Bi(c,o,!0,!0)):(p=Bi(c,i,!1,!0),d=Bi(o,c,!0,!1)),null!=a){return[p,d,rd(l,c)]}return[p,d]},E={a:x,b:v,bias:T,preluActivationWeights:S},_={transposeA:n,transposeB:r,activation:s,leakyreluAlpha:i};if(null==a){return vl((e,t,n)=>{const r=fs.runKernel(wr,E,_);return n([e,t,r]),{value:Di(r,N),gradFunc:k}})(x,v)}return vl((e,t,n,r)=>{const a=fs.runKernel(wr,E,_);return r([e,t,a,n]),{value:Di(a,N),gradFunc:k}})(x,v,T)}}),pd=Object.freeze(Object.defineProperty({__proto__:null,conv2d:od,depthwiseConv2d:ld,matMul:cd},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dd=Is({hammingWindow_:
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e){return Qp(e,.54,.46)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hd=Is({hannWindow_:function(e){return Qp(e,.5,.5)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const md=Is({frame_:function(e,t,n,r=!1,a=0){let s=0;const o=[];for(;s+t<=e.size;)o.push(Li(e,s,t)),s+=n;if(r)for(;s<e.size;){const r=s+t-e.size,i=Fi([Li(e,s,t-r),Yi([r],a)]);o.push(i),s+=n}return 0===o.length?Ep([],[0,t]):Di(Fi(o),[o.length,t])}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fd=Is({stft_:function(e,t,n,r,a=hd){null==r&&(r=Yp(t));const s=md(e,t,n),o=ai(s,a(t));return bp(o,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gd=Is({cropAndResize_:function(e,t,n,r,a="bilinear",s=0){const o=ks(e,"image","cropAndResize"),i=ks(t,"boxes","cropAndResize","float32"),u=ks(n,"boxInd","cropAndResize","int32"),l=i.shape[0];d(4===o.rank,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),d(2===i.rank&&4===i.shape[1],()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${i.shape}.`),d(1===u.rank&&u.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${i.shape}.`),d(2===r.length,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${r.length}.`),d(r[0]>=1&&r[1]>=1,()=>`cropSize must be atleast [1,1], but was ${r}`),d("bilinear"===a||"nearest"===a,()=>`method must be bilinear or nearest, but was ${a}`);const c={image:o,boxes:i,boxInd:u},p={method:a,extrapolationValue:s,cropSize:r};return fs.runKernel(je,c,p)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yd=Is({flipLeftRight_:function(e){const t=ks(e,"image","flipLeftRight","float32");d(4===t.rank,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return fs.runKernel(pt,n,{})}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bd=Is({grayscaleToRGB_:function(e){const t=ks(e,"image","grayscaleToRGB"),n=t.rank-1,r=t.shape[n];d(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),d(1===r,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${r}.`);const a=new Array(t.rank);return a.fill(1,0,n),a[n]=3,el(t,a)}});
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wd=Is({rgbToGrayscale_:function(e){const t=ks(e,"image","RGBToGrayscale"),n=t.rank-1,r=t.shape[n];d(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),d(3===r,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${r}.`);const a=t.dtype,s=Yo(t,"float32"),o=kp([.2989,.587,.114]);let i;switch(t.rank){case 2:i=Au("ij,j->i",s,o);break;case 3:i=Au("ijk,k->ij",s,o);break;case 4:i=Au("ijkl,l->ijk",s,o);break;case 5:i=Au("ijklm,m->ijkl",s,o);break;case 6:i=Au("ijklmn,n->ijklm",s,o);break;default:throw new Error("Not a valid tensor rank.")}return i=Yu(i,-1),Yo(i,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nd=Is({rotateWithOffset_:function(e,t,n=0,r=.5){const a=ks(e,"image","rotateWithOffset","float32");d(4===a.rank,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${a.rank}.`);const s={image:a},o={radians:t,fillValue:n,center:r};return fs.runKernel(br,s,o)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xd(e,t,n,r,a,s){null==r&&(r=.5),null==a&&(a=Number.NEGATIVE_INFINITY),null==s&&(s=0);const o=e.shape[0];return n=Math.min(n,o),d(0<=r&&r<=1,()=>`iouThreshold must be in [0, 1], but was '${r}'`),d(2===e.rank,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),d(4===e.shape[1],()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),d(1===t.rank,()=>"scores must be a 1D tensor"),d(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),d(0<=s&&s<=1,()=>`softNmsSigma must be in [0, 1], but was '${s}'`),{maxOutputSize:n,iouThreshold:r,scoreThreshold:a,softNmsSigma:s}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vd=Is({nonMaxSuppression_:function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY){const s=ks(e,"boxes","nonMaxSuppression","float32"),o=ks(t,"scores","nonMaxSuppression","float32"),i=xd(s,o,n,r,a),u={maxOutputSize:n=i.maxOutputSize,iouThreshold:r=i.iouThreshold,scoreThreshold:a=i.scoreThreshold};return fs.runKernel(nn,{boxes:s,scores:o},u)}});
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Td(e,t,n){const r=function(e,t,n){return function(e,t,n){let r=0,a=e.length,s=0,o=!1;for(;r<a;){s=r+(a-r>>>1);const i=n(t,e[s]);i>0?r=s+1:(a=s,o=!i)}return o?r:-r-1}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(e,t,n||Sd)}(e,t,n),a=r<0?-(r+1):r;e.splice(a,0,t)}function Sd(e,t){return e>t?1:e<t?-1:0}function kd(e,t,n,r,a){return Id(e,t,n,r,a,0)}function Ed(e,t,n,r,a,s){return Id(e,t,n,r,a,0,!1,s,!0)}function _d(e,t,n,r,a,s){return Id(e,t,n,r,a,s,!0)}function Id(e,t,n,r,a,s,o=!1,i=!1,u=!1){const l=[];for(let g=0;g<t.length;g++)t[g]>a&&l.push({score:t[g],boxIndex:g,suppressBeginIndex:0});l.sort($d);const c=s>0?-.5/s:0,p=[],d=[];for(;p.length<n&&l.length>0;){const t=l.pop(),{score:n,boxIndex:s,suppressBeginIndex:o}=t;if(n<a)break;let i=!1;for(let u=p.length-1;u>=o;--u){const n=Ad(e,s,p[u]);if(n>=r){i=!0;break}if(t.score=t.score*Md(r,c,n),t.score<=a)break}t.suppressBeginIndex=p.length,i||(t.score===n?(p.push(s),d.push(t.score)):t.score>a&&Td(l,t,$d))}const h=p.length,m=n-h;i&&m>0&&(p.push(...new Array(m).fill(0)),d.push(...new Array(m).fill(0)));const f={selectedIndices:p};return o&&(f.selectedScores=d),u&&(f.validOutputs=h),f}function Ad(e,t,n){const r=e.subarray(4*t,4*t+4),a=e.subarray(4*n,4*n+4),s=Math.min(r[0],r[2]),o=Math.min(r[1],r[3]),i=Math.max(r[0],r[2]),u=Math.max(r[1],r[3]),l=Math.min(a[0],a[2]),c=Math.min(a[1],a[3]),p=Math.max(a[0],a[2]),d=Math.max(a[1],a[3]),h=(i-s)*(u-o),m=(p-l)*(d-c);if(h<=0||m<=0)return 0;const f=Math.max(s,l),g=Math.max(o,c),y=Math.min(i,p),b=Math.min(u,d),w=Math.max(y-f,0)*Math.max(b-g,0);return w/(h+m-w)}function Md(e,t,n){const r=Math.exp(t*n*n);return n<=e?r:0}function $d(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Od=async function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY){const s=ks(e,"boxes","nonMaxSuppressionAsync"),o=ks(t,"scores","nonMaxSuppressionAsync"),i=xd(s,o,n,r,a);n=i.maxOutputSize,r=i.iouThreshold,a=i.scoreThreshold;const u=await Promise.all([s.data(),o.data()]),l=u[0],c=u[1],{selectedIndices:p}=kd(l,c,n,r,a);return s!==e&&s.dispose(),o!==t&&o.dispose(),kp(p,"int32")};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dd=Is({nonMaxSuppressionWithScore_:function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,s=0){const o=ks(e,"boxes","nonMaxSuppression"),i=ks(t,"scores","nonMaxSuppression"),u=xd(o,i,n,r,a,s),l={boxes:o,scores:i},c={maxOutputSize:n=u.maxOutputSize,iouThreshold:r=u.iouThreshold,scoreThreshold:a=u.scoreThreshold,softNmsSigma:s=u.softNmsSigma},p=fs.runKernel(an,l,c);return{selectedIndices:p[0],selectedScores:p[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cd=async function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,s=0){const o=ks(e,"boxes","nonMaxSuppressionAsync"),i=ks(t,"scores","nonMaxSuppressionAsync"),u=xd(o,i,n,r,a,s);n=u.maxOutputSize,r=u.iouThreshold,a=u.scoreThreshold,s=u.softNmsSigma;const l=await Promise.all([o.data(),i.data()]),c=l[0],p=l[1],{selectedIndices:d,selectedScores:h}=_d(c,p,n,r,a,s);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:kp(d,"int32"),selectedScores:kp(h)}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rd=Is({nonMaxSuppressionPadded_:function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,s=!1){const o=ks(e,"boxes","nonMaxSuppression"),i=ks(t,"scores","nonMaxSuppression"),u=xd(o,i,n,r,a,null),l={boxes:o,scores:i},c={maxOutputSize:u.maxOutputSize,iouThreshold:u.iouThreshold,scoreThreshold:u.scoreThreshold,padToMaxOutputSize:s},p=fs.runKernel(rn,l,c);return{selectedIndices:p[0],validOutputs:p[1]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fd=async function(e,t,n,r=.5,a=Number.NEGATIVE_INFINITY,s=!1){const o=ks(e,"boxes","nonMaxSuppressionAsync"),i=ks(t,"scores","nonMaxSuppressionAsync"),u=xd(o,i,n,r,a,null),l=u.maxOutputSize,c=u.iouThreshold,p=u.scoreThreshold,[d,h]=await Promise.all([o.data(),i.data()]),{selectedIndices:m,validOutputs:f}=Ed(d,h,l,c,p,s);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:kp(m,"int32"),validOutputs:Wu(f,"int32")}};
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bd=Is({resizeBilinear_:function(e,t,n=!1,r=!1){const a=ks(e,"images","resizeBilinear");d(3===a.rank||4===a.rank,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${a.rank}.`),d(2===t.length,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),d(!1===r||!1===n,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let s=a,o=!1;3===a.rank&&(o=!0,s=Di(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const i={images:s},u={alignCorners:n,halfPixelCenters:r,size:t},l=fs.runKernel(Sn,i,u);return o?Di(l,[l.shape[1],l.shape[2],l.shape[3]]):l}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zd=Is({resizeNearestNeighbor_:function(e,t,n=!1,r=!1){const a=ks(e,"images","resizeNearestNeighbor");d(3===a.rank||4===a.rank,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${a.rank}.`),d(2===t.length,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),d("float32"===a.dtype||"int32"===a.dtype,()=>"`images` must have `int32` or `float32` as dtype"),d(!1===r||!1===n,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let s=a,o=!1;3===a.rank&&(o=!0,s=Di(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const i={images:s},u={alignCorners:n,halfPixelCenters:r,size:t},l=fs.runKernel(vn,i,u);return o?Di(l,[l.shape[1],l.shape[2],l.shape[3]]):l}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ld=Is({threshold_:function(e,t="binary",n=!1,r=.5){const a=ks(e,"image","threshold"),s=a.shape[0]*a.shape[1];let o,i,u,l,c=ai(kp([r]),255);if(d(3===a.rank,()=>`Error in threshold: image must be rank 3,but got rank ${a.rank}.`),d(3===a.shape[2]||1===a.shape[2],()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${a.shape[2]}.`),d("int32"===a.dtype||"float32"===a.dtype,()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${a.dtype}.`),d("otsu"===t||"binary"===t,()=>`Method must be binary or otsu, but was ${t}`),3===a.shape[2]){[o,i,u]=yp(a,[1,1,1],-1);const e=ai(o,.2989),t=ai(i,.587),n=ai(u,.114);l=ti(ti(e,t),n)}else l=e;if("otsu"===t){c=function(e,t){let n,r,a,s,o,i,u=kp([-1]),l=kp([0]),c=kp([0]);for(let p=0;p<e.size-1;p++){n=Li(e,0,p+1),r=Li(e,p+1),o=ri(Ku(n),t),i=ri(Ku(r),t);const d=Ku(ai(n,qc(0,n.size)));a=ri(d,Ku(n));const h=Yi(r.shape,n.size),m=ti(qc(0,r.size),h),f=ai(r,m);s=ri(Ku(f),Ku(r));const g=_l(a,s),y=_l(a,s),b=ai(o,i);c=ai(ai(b,g),y);const w=al(c,l);l=ku(w,c,l),u=ku(w,kp([p]),u)}return u}(Ki(Yo(tp(l),"int32"),$s([]),256),s)}const p=n?dl(l,c):al(l,c);return Yo(ai(p,255),"int32")}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pd=Is({transform_:function(e,t,n="nearest",r="constant",a=0,s){const o=ks(e,"image","transform","float32"),i=ks(t,"transforms","transform","float32");d(4===o.rank,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),d(2===i.rank&&(i.shape[0]===o.shape[0]||1===i.shape[0])&&8===i.shape[1],()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),d(null==s||2===s.length,()=>`Error in transform: outputShape must be [height, width] or null, but got ${s}.`);const u={image:o,transforms:i},l={interpolation:n,fillMode:r,fillValue:a,outputShape:s};return fs.runKernel(lr,u,l)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vd=Is({bandPart_:function(e,t,n){const r=ks(e,"a","bandPart");d(r.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${r.rank}.`);const a=r.shape,[s,o]=r.shape.slice(-2);let i,u;"number"==typeof t?(d(t%1==0,()=>`bandPart(): numLower must be an integer, got ${t}.`),d(t<=s,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${s}).`),i=ks(t<0?s:t,"numLower","bandPart")):(d("int32"===t.dtype,()=>"bandPart(): numLower's dtype must be an int32."),i=ku(pl(t,0),s,ql(t,s))),"number"==typeof n?(d(n%1==0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),d(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),u=ks(n<0?o:n,"numUpper","bandPart")):(d("int32"===n.dtype,()=>"bandPart(): numUpper's dtype must be an int32."),u=ku(pl(n,0),o,ql(n,o)));const l=Di(qc(0,s,1,"int32"),[-1,1]),c=qc(0,o,1,"int32"),p=_l(l,c),h=Ml(dl(p,i),sl(p,Sl(u))),m=Ul([s,o],r.dtype);return Di(xp(Pp(Di(r,[-1,s,o])).map(e=>ku(h,e,m))),a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ud=Is({gramSchmidt_:function(e){let t;if(Array.isArray(e)){t=!1,d(null!=e&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const n=e[0].shape[0];for(let t=1;t<e.length;++t)d(e[t].shape[0]===n,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[t].shape[0]} vs. ${n})`)}else t=!0,e=yp(e,e.shape[0],0).map(e=>Np(e,[0]));d(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],r=e;for(let a=0;a<e.length;++a)n.push(fs.tidy(()=>{let e=r[a];if(a>0)for(let t=0;t<a;++t){const r=ai(Ku(ai(n[t],e)),n[t]);e=_l(e,r)}return ri(e,Ju(e,"euclidean"))}));return t?xp(n,0):n}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jd(e,t=!1){return fs.tidy(()=>{d(2===e.shape.length,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],r=e.shape[1];let a=tl(n),s=Qo(e);const o=Ep([[1]],[1,1]);let i=Qo(o);const u=n>=r?r:n;for(let e=0;e<u;++e){const t=s,u=i,l=a;[i,s,a]=fs.tidy(()=>{const t=Li(s,[e,e],[n-e,1]),u=Ju(t),l=Li(s,[e,e],[1,1]),c=ku(al(l,0),Ep([[-1]]),Ep([[1]])),p=_l(l,ai(c,u)),d=ri(t,p);i=1===d.shape[0]?Qo(o):Fi([o,Li(d,[1,0],[d.shape[0]-1,d.shape[1]])],0);const h=Sl(ri(Bi(c,p),u)),m=Li(s,[e,0],[n-e,r]),f=ai(h,i),g=Gp(i);if(0===e)s=_l(m,Bi(f,Bi(g,m)));else{const t=_l(m,Bi(f,Bi(g,m)));s=Fi([Li(s,[0,0],[e,r]),t],0)}const y=Gp(f),b=Li(a,[0,e],[n,a.shape[1]-e]);if(0===e)a=_l(b,Bi(Bi(b,i),y));else{const t=_l(b,Bi(Bi(b,i),y));a=Fi([Li(a,[0,0],[n,e]),t],1)}return[i,s,a]}),js([t,u,l])}return!t&&n>r&&(a=Li(a,[0,0],[n,r]),s=Li(s,[0,0],[r,r])),[a,s]})}const Wd=Is({qr_:function(e,t=!1){if(d(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),2===e.rank)return jd(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((e,t)=>e*t),r=Pp(Di(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),a=[],s=[];r.forEach(e=>{const[n,r]=jd(e,t);a.push(n),s.push(r)});return[Di(xp(a,0),e.shape),Di(xp(s,0),e.shape)]}}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var qd,Gd;(Gd=qd||(qd={}))[Gd.NONE=0]="NONE",Gd[Gd.MEAN=1]="MEAN",Gd[Gd.SUM=2]="SUM",Gd[Gd.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS";const Kd=Is({computeWeightedLoss_:function(e,t,n=qd.SUM_BY_NONZERO_WEIGHTS){const r=ks(e,"losses","computeWeightedLoss");let a=null;null!=t&&(a=ks(t,"weights","computeWeightedLoss"));const s=null==a?r:ai(r,a);if(n===qd.NONE)return s;if(n===qd.SUM)return Ku(s);if(n===qd.MEAN){if(null==a)return Vl(s);{const e=r.size/a.size,t=ri(Ku(s),Ku(a));return e>1?ri(t,Wu(e)):t}}if(n===qd.SUM_BY_NONZERO_WEIGHTS){if(null==a)return ri(Ku(s),Wu(r.size));{const e=ai(a,jl(r.shape)),t=Yo(Ku(Xl(e,Wu(0))),"float32");return ri(Ku(s),t)}}throw Error(`Unknown reduction: ${n}`)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hd=Is({absoluteDifference_:function(e,t,n,r=qd.SUM_BY_NONZERO_WEIGHTS){const a=ks(e,"labels","absoluteDifference"),s=ks(t,"predictions","absoluteDifference");let o=null;null!=n&&(o=ks(n,"weights","absoluteDifference")),h(a.shape,s.shape,"Error in absoluteDifference: ");const i=si(_l(a,s));return Kd(i,o,r)}});const Jd=Is({cosineDistance_:function(e,t,n,r,a=qd.SUM_BY_NONZERO_WEIGHTS){const s=ks(e,"labels","cosineDistance"),o=ks(t,"predictions","cosineDistance");let i=null;null!=r&&(i=ks(r,"weights","cosineDistance")),h(s.shape,o.shape,"Error in cosineDistance: ");const u=Wu(1),l=_l(u,Ku(ai(s,o),n,!0));return Kd(l,i,a)}});const Zd=Is({hingeLoss_:function(e,t,n,r=qd.SUM_BY_NONZERO_WEIGHTS){let a=ks(e,"labels","hingeLoss");const s=ks(t,"predictions","hingeLoss");let o=null;null!=n&&(o=ks(n,"weights","hingeLoss")),h(a.shape,s.shape,"Error in hingeLoss: ");const i=Wu(1);a=_l(ai(Wu(2),a),i);const u=Hc(_l(i,ai(a,s)));return Kd(u,o,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xd=Is({huberLoss_:function(e,t,n,r=1,a=qd.SUM_BY_NONZERO_WEIGHTS){const s=ks(e,"labels","huberLoss"),o=ks(t,"predictions","huberLoss");let i=null;null!=n&&(i=ks(n,"weights","huberLoss")),h(s.shape,o.shape,"Error in huberLoss: ");const u=Wu(r),l=si(_l(o,s)),c=ql(l,u),p=_l(l,c),d=ti(ai(Wu(.5),Gu(c)),ai(u,p));return Kd(d,i,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yd=Is({logLoss_:function(e,t,n,r=1e-7,a=qd.SUM_BY_NONZERO_WEIGHTS){const s=ks(e,"labels","logLoss"),o=ks(t,"predictions","logLoss");let i=null;null!=n&&(i=ks(n,"weights","logLoss")),h(s.shape,o.shape,"Error in logLoss: ");const u=Wu(1),l=Wu(r),c=Sl(ai(s,fl(ti(o,l)))),p=ai(_l(u,s),fl(ti(_l(u,o),l))),d=_l(c,p);return Kd(d,i,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qd=Is({meanSquaredError_:function(e,t,n,r=qd.SUM_BY_NONZERO_WEIGHTS){const a=ks(e,"labels","meanSquaredError"),s=ks(t,"predictions","meanSquaredError");let o=null;null!=n&&(o=ks(n,"weights","meanSquaredError")),h(a.shape,s.shape,"Error in meanSquaredError: ");const i=wp(a,s);return Kd(i,o,r)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const eh=Is({sigmoidCrossEntropy_:function(e,t,n,r=0,a=qd.SUM_BY_NONZERO_WEIGHTS){let s=ks(e,"multiClassLabels","sigmoidCrossEntropy");const o=ks(t,"logits","sigmoidCrossEntropy");let i=null;if(null!=n&&(i=ks(n,"weights","sigmoidCrossEntropy")),h(s.shape,o.shape,"Error in sigmoidCrossEntropy: "),r>0){const e=Wu(r),t=Wu(1),n=Wu(.5);s=ti(ai(s,_l(t,e)),ai(n,e))}const u=function(e,t){const n=ks(e,"labels","sigmoidCrossEntropyWithLogits"),r=ks(t,"logits","sigmoidCrossEntropyWithLogits");h(n.shape,r.shape,"Error in sigmoidCrossEntropyWithLogits: ");const a=Hc(r),s=ai(r,n),o=gl(Xu(Sl(si(r))));return ti(_l(a,s),o)}(s,o);return Kd(u,i,a)}});
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const th=Is({softmaxCrossEntropy_:function(e,t,n,r=0,a=qd.SUM_BY_NONZERO_WEIGHTS){let s=ks(e,"onehotLabels","softmaxCrossEntropy");const o=ks(t,"logits","softmaxCrossEntropy");let i=null;if(null!=n&&(i=ks(n,"weights","softmaxCrossEntropy")),h(s.shape,o.shape,"Error in softmaxCrossEntropy: "),r>0){const e=Wu(r),t=Wu(1),n=Wu(s.shape[1]);s=ti(ai(s,_l(t,e)),ri(e,n))}const u=function(e,t,n=-1){if(-1===n&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return vl((e,t,r)=>{const a=Al(t,[n],!0),s=_l(Yo(t,"float32"),a);r([e,s]);const o=Sl(ai(s,e));return{value:Ku(o,[n]),gradFunc:(e,t)=>{const[r,a]=t,s=Fu(e.shape,[n]);return[ai(Di(e,s),_l(Yo(r,"float32"),Xu(a))),ai(Di(e,s),_l(Xu(a),Yo(r,"float32")))]}}})(e,t)}(s,o);return Kd(u,i,a)}});
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nh={fft:mp,ifft:fp,rfft:bp,irfft:gp},rh={hammingWindow:dd,hannWindow:hd,frame:md,stft:fd},ah={flipLeftRight:yd,grayscaleToRGB:bd,resizeNearestNeighbor:zd,resizeBilinear:Bd,rgbToGrayscale:wd,rotateWithOffset:Nd,cropAndResize:gd,nonMaxSuppression:vd,nonMaxSuppressionAsync:Od,nonMaxSuppressionWithScore:Dd,nonMaxSuppressionWithScoreAsync:Cd,nonMaxSuppressionPadded:Rd,nonMaxSuppressionPaddedAsync:Fd,threshold:Ld,transform:Pd},sh={bandPart:Vd,gramSchmidt:Ud,qr:Wd},oh={absoluteDifference:Hd,computeWeightedLoss:Kd,cosineDistance:Jd,hingeLoss:Zd,huberLoss:Xd,logLoss:Yd,meanSquaredError:Qd,sigmoidCrossEntropy:eh,softmaxCrossEntropy:th},ih={sparseFillEmptyRows:Is({sparseFillEmptyRows_:function(e,t,n,r){const a=ks(e,"indices","sparseFillEmptyRows","int32"),s=ks(t,"values","sparseFillEmptyRows"),o=ks(n,"denseShape","sparseFillEmptyRows","int32"),i=ks(r,"defaultValue","sparseFillEmptyRows",s.dtype);if(2!==a.rank)throw new Error(`Indices should be Tensor2D but received shape\n        ${a.shape}`);if(1!==s.rank)throw new Error(`Values should be Tensor1D but received shape ${s.shape}`);if(1!==o.rank)throw new Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(0!==i.rank)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const u={indices:a,values:s,denseShape:o,defaultValue:i},l=fs.runKernel(Gn,u);return{outputIndices:l[0],outputValues:l[1],emptyRowIndicator:l[2],reverseIndexMap:l[3]}}}),sparseReshape:Is({sparseReshape_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n){const r=ks(e,"inputIndices","sparseReshape","int32"),a=ks(t,"inputShape","sparseReshape","int32"),s=ks(n,"newShape","sparseReshape","int32");if(2!==r.rank)throw new Error(`Input indices should be Tensor2D but received shape\n        ${r.shape}`);if(1!==a.rank)throw new Error(`Input shape should be Tensor1D but received shape ${a.shape}`);if(1!==s.rank)throw new Error(`New shape should be Tensor1D but received shape ${s.shape}`);const o={inputIndices:r,inputShape:a,newShape:s},i=fs.runKernel(Kn,o);return{outputIndices:i[0],outputShape:i[1]}}}),sparseSegmentMean:Is({sparseSegmentMean_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n){const r=ks(e,"data","sparseSegmentMean"),a=ks(t,"indices","sparseSegmentMean","int32"),s=ks(n,"segmentIds","sparseSegmentMean","int32");if(r.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(1!==a.rank)throw new Error(`Indices should be Tensor1D but received shape\n          ${a.shape}`);if(1!==s.rank)throw new Error(`Segment ids should be Tensor1D but received shape\n          ${s.shape}`);const o={data:r,indices:a,segmentIds:s};return fs.runKernel(Hn,o)}}),sparseSegmentSum:Is({sparseSegmentSum_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n){const r=ks(e,"data","sparseSegmentSum"),a=ks(t,"indices","sparseSegmentSum","int32"),s=ks(n,"segmentIds","sparseSegmentSum","int32");if(r.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(1!==a.rank)throw new Error(`Indices should be Tensor1D but received shape\n         ${a.shape}`);if(1!==s.rank)throw new Error(`Segment ids should be Tensor1D but received shape\n         ${s.shape}`);const o={data:r,indices:a,segmentIds:s};return fs.runKernel(Jn,o)}})},uh={stringNGrams:Is({stringNGrams_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r,a,s,o,i){const u=ks(e,"data","stringNGrams","string");if("string"!==u.dtype)throw new Error("Data must be of datatype string");if(1!==u.shape.length)throw new Error(`Data must be a vector, saw: ${u.shape}`);const l=ks(t,"dataSplits","stringNGrams");if("int32"!==l.dtype)throw new Error("Data splits must be of datatype int32");const c={separator:n,nGramWidths:r,leftPad:a,rightPad:s,padWidth:o,preserveShortSequences:i},p={data:u,dataSplits:l},d=fs.runKernel(tr,p,c);return{nGrams:d[0],nGramsSplits:d[1]}}}),stringSplit:Is({stringSplit_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n=!0){const r=ks(e,"input","stringSplit","string"),a=ks(t,"delimiter","stringSplit","string");if(1!==r.rank)throw new Error(`Input should be Tensor1D but received shape ${r.shape}`);if(0!==a.rank)throw new Error(`Delimiter should be a scalar but received shape ${a.shape}`);const s={skipEmpty:n},o={input:r,delimiter:a},i=fs.runKernel(nr,o,s);return{indices:i[0],values:i[1],shape:i[2]}}}),stringToHashBucketFast:Is({stringToHashBucketFast_:
/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t){const n=ks(e,"input","stringToHashBucketFast","string"),r={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const a={input:n};return fs.runKernel(rr,a,r)}}),staticRegexReplace:Is({staticRegexReplace_:
/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function(e,t,n,r=!0){const a=ks(e,"input","staticRegexReplace","string"),s={pattern:t,rewrite:n,replaceGlobal:r};return fs.runKernel(Qn,{x:a},s)}})},lh=new Map,ch=new Map;
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ph{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}}class dh{constructor(){this.classNameMap={}}static getMap(){return null==dh.instance&&(dh.instance=new dh),dh.instance}static register(e){dh.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function hh(e,t,n){d(null!=e.className,()=>"Class being registered does not have the static className property defined."),d("string"==typeof e.className,()=>"className is required to be a string, but got type "+typeof e.className),d(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),void 0===t&&(t="Custom"),void 0===n&&(n=e.className);const r=t+">"+n;return dh.register(e),lh.set(r,e),ch.set(e,r),e}const mh=Object.freeze(Object.defineProperty({__proto__:null,Serializable:ph,SerializationMap:dh,getRegisteredName:function(e){return ch.has(e)?ch.get(e):e.className},registerClass:hh},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fh extends ph{minimize(e,t=!1,n){const{value:r,grads:a}=this.computeGradients(e,n);if(null!=n){const e=n.map(e=>({name:e.name,tensor:a[e.name]}));this.applyGradients(e)}else this.applyGradients(a);return js(a),t?r:(r.dispose(),null)}get iterations(){return null==this.iterations_&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return xl(e,t)}dispose(){null!=this.iterations_&&js(this.iterations_)}async saveIterations(){return null==this.iterations_&&(this.iterations_=0),{name:"iter",tensor:Wu(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(fh,Symbol.hasInstance,{value:e=>null!=e.minimize&&null!=e.computeGradients&&null!=e.applyGradients});
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class gh extends fh{static get className(){return"Adadelta"}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],null==n&&(this.epsilon=fs.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=fs.registeredVariables[t],a=!1;null==this.accumulatedGrads[n]&&(this.accumulatedGrads[n]={originalName:`${t}/accum_grad`,variable:Us(()=>Eu(r).variable(a))}),null==this.accumulatedUpdates[n]&&(this.accumulatedUpdates[n]={originalName:`${t}/accum_var`,variable:Us(()=>Eu(r).variable(a))});const s=Array.isArray(e)?e[n].tensor:e[t];if(null==s)return;const o=this.accumulatedGrads[n].variable,i=this.accumulatedUpdates[n].variable;Us(()=>{const e=ti(ai(o,this.rho),ai(Gu(s),1-this.rho)),t=ai(ri(qu(ti(i,this.epsilon)),qu(ti(o,this.epsilon))),s),n=ti(ai(i,this.rho),ai(Gu(t),1-this.rho));o.assign(e),i.assign(n);const a=ti(ai(t,-this.learningRate),r);r.assign(a)})}),this.incrementIterations()}dispose(){null!=this.accumulatedUpdates&&(js(this.accumulatedGrads.map(e=>e.variable)),js(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){const e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){const t=(e=await this.extractIterations(e)).length/2,n=!1;this.accumulatedGrads=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.accumulatedUpdates=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class yh extends fh{static get className(){return"Adagrad"}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=fs.registeredVariables[t];if(null==this.accumulatedGrads[n]){const e=!1;this.accumulatedGrads[n]={originalName:`${t}/accumulator`,variable:Us(()=>Yi(r.shape,this.initialAccumulatorValue).variable(e))}}const a=Array.isArray(e)?e[n].tensor:e[t];if(null==a)return;const s=this.accumulatedGrads[n].variable;Us(()=>{const e=ti(s,Gu(a));s.assign(e);const t=ti(ai(ri(a,qu(ti(e,fs.backend.epsilon()))),-this.learningRate),r);r.assign(t)})}),this.incrementIterations()}dispose(){null!=this.accumulatedGrads&&js(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);this.accumulatedGrads=e.map(e=>({originalName:e.name,variable:e.tensor.variable(false)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class bh extends fh{static get className(){return"Adam"}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],Us(()=>{this.accBeta1=Wu(t).variable(),this.accBeta2=Wu(n).variable()}),null==r&&(this.epsilon=fs.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);Us(()=>{const n=_l(1,this.accBeta1),r=_l(1,this.accBeta2);t.forEach((t,a)=>{const s=fs.registeredVariables[t],o=!1;null==this.accumulatedFirstMoment[a]&&(this.accumulatedFirstMoment[a]={originalName:`${t}/m`,variable:Us(()=>Eu(s).variable(o))}),null==this.accumulatedSecondMoment[a]&&(this.accumulatedSecondMoment[a]={originalName:`${t}/v`,variable:Us(()=>Eu(s).variable(o))});const i=Array.isArray(e)?e[a].tensor:e[t];if(null==i)return;const u=this.accumulatedFirstMoment[a].variable,l=this.accumulatedSecondMoment[a].variable,c=ti(ai(u,this.beta1),ai(i,1-this.beta1)),p=ti(ai(l,this.beta2),ai(Gu(i),1-this.beta2)),d=ri(c,n),h=ri(p,r);u.assign(c),l.assign(p);const m=ti(ai(ri(d,ti(qu(h),this.epsilon)),-this.learningRate),s);s.assign(m)}),this.accBeta1.assign(ai(this.accBeta1,this.beta1)),this.accBeta2.assign(ai(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),null!=this.accumulatedFirstMoment&&js(this.accumulatedFirstMoment.map(e=>e.variable)),null!=this.accumulatedSecondMoment&&js(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e),Us(()=>{this.accBeta1.assign(ju(this.beta1,this.iterations_+1)),this.accBeta2.assign(ju(this.beta2,this.iterations_+1))});const t=e.length/2,n=!1;this.accumulatedFirstMoment=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.accumulatedSecondMoment=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wh extends fh{static get className(){return"Adamax"}constructor(e,t,n,r=null,a=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=a,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],Us(()=>{this.iteration=Wu(0).variable(),this.accBeta1=Wu(t).variable()}),null==r&&(this.epsilon=fs.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(e=>e.name):Object.keys(e);Us(()=>{const n=_l(1,this.accBeta1),r=ri(-this.learningRate,ti(ai(this.iteration,this.decay),1));t.forEach((t,a)=>{const s=fs.registeredVariables[t],o=!1;null==this.accumulatedFirstMoment[a]&&(this.accumulatedFirstMoment[a]={originalName:`${t}/m`,variable:Eu(s).variable(o)}),null==this.accumulatedWeightedInfNorm[a]&&(this.accumulatedWeightedInfNorm[a]={originalName:`${t}/v`,variable:Eu(s).variable(o)});const i=Array.isArray(e)?e[a].tensor:e[t];if(null==i)return;const u=this.accumulatedFirstMoment[a].variable,l=this.accumulatedWeightedInfNorm[a].variable,c=ti(ai(u,this.beta1),ai(i,1-this.beta1)),p=ai(l,this.beta2),d=si(i),h=Pl(p,d);u.assign(c),l.assign(h);const m=ti(ai(ri(r,n),ri(c,ti(h,this.epsilon))),s);s.assign(m)}),this.iteration.assign(ti(this.iteration,1)),this.accBeta1.assign(ai(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),null!=this.accumulatedFirstMoment&&js(this.accumulatedFirstMoment.map(e=>e.variable)),null!=this.accumulatedWeightedInfNorm&&js(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Nh extends fh{static get className(){return"SGD"}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=Array.isArray(e)?e[n].tensor:e[t];if(null==r)return;const a=fs.registeredVariables[t];Us(()=>{const e=ti(ai(this.c,r),a);a.assign(e)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,null!=this.c&&this.c.dispose(),this.c=Ws(Wu(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(0!==(e=await this.extractIterations(e)).length)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class xh extends Nh{static get className(){return"Momentum"}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=Wu(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=fs.registeredVariables[t];if(null==this.accumulations[n]){const e=!1;this.accumulations[n]={originalName:`${t}/momentum`,variable:Us(()=>Eu(r).variable(e))}}const a=this.accumulations[n].variable,s=Array.isArray(e)?e[n].tensor:e[t];null!=s&&Us(()=>{let e;const t=ti(ai(this.m,a),s);e=this.useNesterov?ti(ai(this.c,ti(s,ai(t,this.m))),r):ti(ai(this.c,t),r),a.assign(t),r.assign(e)})}),this.incrementIterations()}dispose(){this.m.dispose(),null!=this.accumulations&&js(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);this.accumulations=e.map(e=>({originalName:e.name,variable:e.tensor.variable(false)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class vh extends fh{static get className(){return"RMSProp"}constructor(e,t=.9,n=0,r=null,a=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=a,null==r&&(this.epsilon=fs.backend.epsilon()),null==e)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){(Array.isArray(e)?e.map(e=>e.name):Object.keys(e)).forEach((t,n)=>{const r=fs.registeredVariables[t],a=!1;null==this.accumulatedMeanSquares[n]&&(this.accumulatedMeanSquares[n]={originalName:`${t}/rms`,variable:Us(()=>Eu(r).variable(a))}),null==this.accumulatedMoments[n]&&(this.accumulatedMoments[n]={originalName:`${t}/momentum`,variable:Us(()=>Eu(r).variable(a))}),null==this.accumulatedMeanGrads[n]&&this.centered&&(this.accumulatedMeanGrads[n]={originalName:`${t}/mg`,variable:Us(()=>Eu(r).variable(a))});const s=Array.isArray(e)?e[n].tensor:e[t];if(null==s)return;const o=this.accumulatedMeanSquares[n].variable,i=this.accumulatedMoments[n].variable;Us(()=>{const e=ti(ai(o,this.decay),ai(Gu(s),1-this.decay));if(this.centered){const t=this.accumulatedMeanGrads[n].variable,a=ti(ai(t,this.decay),ai(s,1-this.decay)),u=ri(ai(s,this.learningRate),qu(_l(e,ti(Gu(a),this.epsilon)))),l=ti(ai(i,this.momentum),u);o.assign(e),t.assign(a),i.assign(l);const c=_l(r,l);r.assign(c)}else{const e=ti(ai(o,this.decay),ai(Gu(s),1-this.decay)),t=ti(ai(i,this.momentum),ri(ai(s,this.learningRate),qu(ti(e,this.epsilon))));o.assign(e),i.assign(t);const n=_l(r,t);r.assign(n)}})}),this.incrementIterations()}dispose(){null!=this.accumulatedMeanSquares&&js(this.accumulatedMeanSquares.map(e=>e.variable)),null!=this.accumulatedMeanGrads&&this.centered&&js(this.accumulatedMeanGrads.map(e=>e.variable)),null!=this.accumulatedMoments&&js(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=this.centered?e.length/3:e.length/2,n=!1;this.accumulatedMeanSquares=e.slice(0,t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.accumulatedMoments=e.slice(t,2*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})),this.centered&&(this.accumulatedMeanGrads=e.slice(2*t,3*t).map(e=>({originalName:e.name,variable:e.tensor.variable(n)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}}
/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Th=[gh,yh,bh,wh,xh,vh,Nh];function Sh(e){return new Promise(e=>setTimeout(e)).then(e)}class kh{constructor(e){if(!Y().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(kh.URL_SCHEME)&&(e=e.slice(kh.URL_SCHEME.length)),null!=e&&0!==e.length||(e="model"),this.modelJsonFileName=e+".json",this.weightDataFileName=e+".weights.bin"}async save(e){if("undefined"==typeof document)throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=Ds.join(e.weightData),n=window.URL.createObjectURL(new Blob([t],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const t=mo(e,[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}]),r=window.URL.createObjectURL(new Blob([JSON.stringify(t)],{type:"application/json"})),a=null==this.modelJsonAnchor?document.createElement("a"):this.modelJsonAnchor;if(a.download=this.modelJsonFileName,a.href=r,await Sh(()=>a.dispatchEvent(new MouseEvent("click"))),null!=e.weightData){const e=null==this.weightDataAnchor?document.createElement("a"):this.weightDataAnchor;e.download=this.weightDataFileName,e.href=n,await Sh(()=>e.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:yo(e)}}}}kh.URL_SCHEME="downloads://";class Eh{constructor(e){if(null==e||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{const n=new FileReader;n.onload=n=>{const r=JSON.parse(n.target.result),a=r.modelTopology;if(null==a)return void t(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));if(null==r.weightsManifest)return void t(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));if(0===this.weightsFiles.length)return void e({modelTopology:a});const s=go(r,e=>this.loadWeights(e));e(s)},n.onerror=e=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),n.readAsText(this.jsonFile)})}loadWeights(e){const t=[],n=[];for(const s of e)t.push(...s.weights),n.push(...s.paths);const r=this.checkManifestAndWeightFiles(e),a=n.map(e=>this.loadWeightsFile(e,r[e]));return Promise.all(a).then(e=>[t,e])}loadWeightsFile(e,t){return new Promise((n,r)=>{const a=new FileReader;a.onload=e=>{const t=e.target.result;n(t)},a.onerror=t=>r(`Failed to weights data from file of path '${e}'.`),a.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){const t=[],n=this.weightsFiles.map(e=>ho(e.name)),r={};for(const a of e)a.paths.forEach(e=>{const a=ho(e);if(-1!==t.indexOf(a))throw new Error(`Duplicate file basename found in weights manifest: '${a}'`);if(t.push(a),-1===n.indexOf(a))throw new Error(`Weight file with basename '${a}' is not provided.`);r[e]=this.weightsFiles[n.indexOf(a)]});if(t.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function _h(e,t,n,r){var a,s,o;d(null!=(a=e)&&Array.isArray(a)&&a.length>0,()=>"promises must be a none empty array"),o=r=null==r?1:r,d((s=n=null==n?0:n)>=0&&s<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${s}`),d(o>=0&&o<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${o}`),d(o>=s,()=>`startFraction must be no more than endFraction, but got startFraction ${s} and endFraction ${o}`);let i=0;return Promise.all(e.map(a=>(a.then(a=>{const s=n+ ++i/e.length*(r-n);return t(s),a}),a)))}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Ih(e,t){null==t&&(t={});const n=null==t.fetchFunc?Y().platform.fetch:t.fetchFunc,r=e.map(e=>n(e,t.requestInit,{isBinary:!0})),a=(null==t.onProgress?await Promise.all(r):await _h(r,t.onProgress,0,.5)).map(e=>e.arrayBuffer());return null==t.onProgress?await Promise.all(a):await _h(a,t.onProgress,.5,1)}async function Ah(e,t="",n,r){return Mh(e=>Ih(e,{requestInit:r}))(e,t,n)}function Mh(e){return async(t,n="",r)=>{const a=t.map(()=>!1),s={},o=null!=r?r.map(()=>!1):[],i=[];if(t.forEach((e,t)=>{let n=0;e.weights.forEach(e=>{const u="quantization"in e?e.quantization.dtype:e.dtype,l=Os[u]*f(e.shape),c=()=>{a[t]=!0,null==s[t]&&(s[t]=[]),s[t].push({manifestEntry:e,groupOffset:n,sizeBytes:l})};null!=r?r.forEach((t,n)=>{t===e.name&&(c(),o[n]=!0)}):c(),i.push(e.name),n+=l})}),!o.every(e=>e)){const e=r.filter((e,t)=>!o[t]);throw new Error(`Could not find weights in manifest with names: ${e.join(", ")}. \nManifest JSON has weights with names: ${i.join(", ")}.`)}const u=a.reduce((e,t,n)=>(t&&e.push(n),e),[]),l=[];u.forEach(e=>{t[e].paths.forEach(e=>{const t=n+(n.endsWith("/")?"":"/")+e;l.push(t)})});const c=await e(l),p={};let d=0;return u.forEach(e=>{const n=t[e].paths.length,r=new Ds(c.slice(d,d+n));s[e].forEach(e=>{const t=no(r.slice(e.groupOffset,e.groupOffset+e.sizeBytes),[e.manifestEntry]);for(const n in t)p[n]=t[n]}),d+=n}),p}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */wo.registerSaveRouter(e=>Y().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(kh.URL_SCHEME)?function(e="model"){return new kh(e)}(e.slice(kh.URL_SCHEME.length)):null);class $h{constructor(e,t){if(this.DEFAULT_METHOD="POST",null==t&&(t={}),this.weightPathPrefix=t.weightPathPrefix,this.weightUrlConverter=t.weightUrlConverter,null!=t.fetchFunc?(d("function"==typeof t.fetchFunc,()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc):this.fetch=Y().platform.fetch,d(null!=e&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&d(2===e.length,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,null!=t.requestInit&&null!=t.requestInit.body)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=t.requestInit||{},this.loadOptions=t}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;const n=mo(e,[{paths:["./model.weights.bin"],weights:e.weightSpecs}]);if(t.body.append("model.json",new Blob([JSON.stringify(n)],{type:"application/json"}),"model.json"),null!=e.weightData){const n=Ds.join(e.weightData);t.body.append("model.weights.bin",new Blob([n],{type:"application/octet-stream"}),"model.weights.bin")}const r=await this.fetch(this.path,t);if(r.ok)return{modelArtifactsInfo:yo(e),responses:[r]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${r.status}.`)}async loadModelJSON(){const e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw new Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch(vf){let t=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?t+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":t+=" Please make sure the server is serving valid JSON for this request.",new Error(t)}const n=t.modelTopology,r=t.weightsManifest;if(null==n&&null==r)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return t}async load(){if(this.loadOptions.streamWeights)return this.loadStream();return go(await this.loadModelJSON(),e=>this.loadWeights(e))}async loadStream(){const e=await this.loadModelJSON(),t=await this.getWeightUrls(e.weightsManifest),n=bo(e.weightsManifest);return Object.assign(Object.assign({},e),{weightSpecs:n,getWeightStream:()=>function(e,t){var n;const r=null==t.fetchFunc?Y().platform.fetch:t.fetchFunc;let a,s=0;return null===(n=t.onProgress)||void 0===n||n.call(t,0),new ReadableStream({pull:async n=>{for(var o;s<e.length;){if(!a){const n=(await r(e[s],t.requestInit,{isBinary:!0})).body;a=n.getReader()}const{done:i,value:u}=await a.read();if(!i)return void n.enqueue(u);s++,a=void 0,null===(o=t.onProgress)||void 0===o||o.call(t,s/e.length)}n.close()}})}(t,this.loadOptions)})}async getWeightUrls(e){const t=Array.isArray(this.path)?this.path[1]:this.path,[n,r]=function(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),r=e.substring(0,t),a=n>t?e.substring(n):"";return[r+"/",a]}(t),a=this.weightPathPrefix||n,s=[],o=[];for(const i of e)for(const e of i.paths)null!=this.weightUrlConverter?o.push(this.weightUrlConverter(e)):s.push(a+e+r);return this.weightUrlConverter&&s.push(...await Promise.all(o)),s}async loadWeights(e){const t=await this.getWeightUrls(e);return[bo(e),await Ih(t,this.loadOptions)]}}function Oh(e){return null!=e.match($h.URL_SCHEME_REGEX)}$h.URL_SCHEME_REGEX=/^https?:\/\//;const Dh=(e,t)=>{if("undefined"==typeof fetch&&(null==t||null==t.fetchFunc))return null;{let n=!0;if(n=Array.isArray(e)?e.every(e=>Oh(e)):Oh(e),n)return Ch(e,t)}return null};function Ch(e,t){return new $h(e,t)}function Rh(e,t){return Ch(e,t)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */wo.registerSaveRouter(Dh),wo.registerLoadRouter(Dh);class Fh{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class Bh{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class zh{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function Lh(e,t,n,r){if(1===arguments.length){const t=null!=e.modelTopology||null!=e.weightSpecs;return new Fh(t?e:{modelTopology:e})}return new Fh({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:r})}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const Ph=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:Ds,browserFiles:function(e){return new Eh(e)},browserHTTPRequest:Rh,concatenateArrayBuffers:po,copyModel:async function(e,t){return Go(e,t,!1)},decodeWeights:no,decodeWeightsStream:io,encodeWeights:to,fromMemory:function(e,t,n,r){return new zh(Lh(...arguments))},fromMemorySync:Lh,getLoadHandlers:xo,getModelArtifactsForJSON:go,getModelArtifactsForJSONSync:fo,getModelArtifactsInfoForJSON:yo,getSaveHandlers:No,getWeightSpecs:bo,http:Ch,isHTTPScheme:Oh,listModels:async function(){const e=Wo.getSchemes(),t={};for(const n of e){const e=await Wo.getManager(n).listModels();for(const r in e){t[n+jo+r]=e[r]}}return t},loadWeights:Ah,moveModel:async function(e,t){return Go(e,t,!0)},registerLoadRouter:e=>wo.registerLoadRouter(e),registerSaveRouter:e=>wo.registerSaveRouter(e),removeModel:async function(e){const t=qo(e);return Wo.getManager(t.scheme).removeModel(t.path)},weightsLoaderFactory:Mh,withSaveHandler:function(e){return new Bh(e)},withSaveHandlerSync:function(e){return new Bh(e)}},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Vh,Uh=!1;function jh(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(null==e)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,r=!1,a=!1,s=!1,o=!1,i=!1;if(e.data instanceof Uint8Array)n=!0;else if("undefined"!=typeof ImageData&&e instanceof ImageData)r=!0;else if("undefined"!=typeof HTMLVideoElement&&e instanceof HTMLVideoElement)a=!0;else if("undefined"!=typeof HTMLImageElement&&e instanceof HTMLImageElement)s=!0;else if(null!=e.getContext)o=!0;else{if(!("undefined"!=typeof ImageBitmap&&e instanceof ImageBitmap))throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);i=!0}if(null!=Er(yr,fs.backendName)){const n={pixels:e},r={numChannels:t};return fs.runKernel(yr,n,r)}const[u,l]=a?[e.videoWidth,e.videoHeight]:[e.width,e.height];let c,p;if(o)c=e.getContext("2d").getImageData(0,0,u,l).data;else if(r||n)c=e.data;else if(s||a||i){if(null==Vh)if("undefined"==typeof document){if("undefined"==typeof OffscreenCanvas||"undefined"==typeof OffscreenCanvasRenderingContext2D)throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");Vh=new OffscreenCanvas(1,1).getContext("2d")}else Vh=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});Vh.canvas.width=u,Vh.canvas.height=l,Vh.drawImage(e,0,0,u,l),c=Vh.getImageData(0,0,u,l).data}if(4===t)p=new Int32Array(c);else{const e=u*l;p=new Int32Array(e*t);for(let n=0;n<e;n++)for(let e=0;e<t;++e)p[n*t+e]=c[4*n+e]}return _p(p,[l,u,t],"int32")}function Wh(e){return"undefined"!=typeof window&&"undefined"!=typeof ImageBitmap&&window.hasOwnProperty("createImageBitmap")&&!(e instanceof ImageBitmap)&&function(e){return null!=e&&0!==e.width&&0!==e.height}(e)&&!function(e){return null!=e&&e.data instanceof Uint8Array}(e)}function qh(e){if(2!==e.rank&&3!==e.rank)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${e.rank}.`);const t=2===e.rank?1:e.shape[2];if(t>4||2===t)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${t}`);if("float32"!==e.dtype&&"int32"!==e.dtype)throw new Error(`Unsupported type for toPixels: ${e.dtype}. Please use float32 or int32 tensors.`)}const Gh=Is({fromPixels_:jh}),Kh=Object.freeze(Object.defineProperty({__proto__:null,draw:function(e,t,n){let r=ks(e,"img","draw");if(!(e instanceof Va)){const e=r;r=Yo(e,"int32"),e.dispose()}qh(r),function(e){const t=(null==e?void 0:e.alpha)||1;if(t>1||t<0)throw new Error(`Alpha value ${t} is suppoed to be in range [0 - 1].`)}(null==n?void 0:n.imageOptions);const a={image:r},s={canvas:t,options:n};fs.runKernel(Qe,a,s)},fromPixels:Gh,fromPixelsAsync:async function(e,t=3){let n=null;if(Y().getBool("WRAP_TO_IMAGEBITMAP")&&Wh(e)){let t;try{t=await createImageBitmap(e,{premultiplyAlpha:"none"})}catch(vf){t=null}n=null!=t&&t.width===e.width&&t.height===e.height?t:e}else n=e;return jh(n,t)},toPixels:async function(e,t){let n=ks(e,"img","toPixels");if(!(e instanceof Va)){const e=n;n=Yo(e,"int32"),e.dispose()}qh(n);const[r,a]=n.shape.slice(0,2),s=2===n.rank?1:n.shape[2],o=await n.data(),i="float32"===n.dtype?255:1,u=new Uint8ClampedArray(a*r*4);for(let l=0;l<r*a;++l){const e=[0,0,0,255];for(let r=0;r<s;r++){const t=o[l*s+r];if("float32"===n.dtype){if(t<0||t>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${t}.`)}else if("int32"===n.dtype&&(t<0||t>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${t}.`);1===s?(e[0]=t*i,e[1]=t*i,e[2]=t*i):e[r]=t*i}const t=4*l;u[t+0]=Math.round(e[0]),u[t+1]=Math.round(e[1]),u[t+2]=Math.round(e[2]),u[t+3]=Math.round(e[3])}if(null!=t){if(!Uh){null!=Er(Qe,fs.backendName)&&(Uh=!0)}t.width=a,t.height=r;const e=t.getContext("2d"),n=new ImageData(u,a,r);e.putImageData(n,0,0)}return n!==e&&n.dispose(),u}},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
!function(){for(const e of Th)hh(e)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */();
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */
var Hh,Jh,Zh,Xh,Yh;Y().registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,e=>{}),(Jh=Hh||(Hh={}))[Jh.DT_INVALID=0]="DT_INVALID",Jh[Jh.DT_FLOAT=1]="DT_FLOAT",Jh[Jh.DT_DOUBLE=2]="DT_DOUBLE",Jh[Jh.DT_INT32=3]="DT_INT32",Jh[Jh.DT_UINT8=4]="DT_UINT8",Jh[Jh.DT_INT16=5]="DT_INT16",Jh[Jh.DT_INT8=6]="DT_INT8",Jh[Jh.DT_STRING=7]="DT_STRING",Jh[Jh.DT_COMPLEX64=8]="DT_COMPLEX64",Jh[Jh.DT_INT64=9]="DT_INT64",Jh[Jh.DT_BOOL=10]="DT_BOOL",Jh[Jh.DT_QINT8=11]="DT_QINT8",Jh[Jh.DT_QUINT8=12]="DT_QUINT8",Jh[Jh.DT_QINT32=13]="DT_QINT32",Jh[Jh.DT_BFLOAT16=14]="DT_BFLOAT16",Jh[Jh.DT_QINT16=15]="DT_QINT16",Jh[Jh.DT_QUINT16=16]="DT_QUINT16",Jh[Jh.DT_UINT16=17]="DT_UINT16",Jh[Jh.DT_COMPLEX128=18]="DT_COMPLEX128",Jh[Jh.DT_HALF=19]="DT_HALF",Jh[Jh.DT_RESOURCE=20]="DT_RESOURCE",Jh[Jh.DT_VARIANT=21]="DT_VARIANT",Jh[Jh.DT_UINT32=22]="DT_UINT32",Jh[Jh.DT_UINT64=23]="DT_UINT64",Jh[Jh.DT_FLOAT_REF=101]="DT_FLOAT_REF",Jh[Jh.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",Jh[Jh.DT_INT32_REF=103]="DT_INT32_REF",Jh[Jh.DT_UINT8_REF=104]="DT_UINT8_REF",Jh[Jh.DT_INT16_REF=105]="DT_INT16_REF",Jh[Jh.DT_INT8_REF=106]="DT_INT8_REF",Jh[Jh.DT_STRING_REF=107]="DT_STRING_REF",Jh[Jh.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",Jh[Jh.DT_INT64_REF=109]="DT_INT64_REF",Jh[Jh.DT_BOOL_REF=110]="DT_BOOL_REF",Jh[Jh.DT_QINT8_REF=111]="DT_QINT8_REF",Jh[Jh.DT_QUINT8_REF=112]="DT_QUINT8_REF",Jh[Jh.DT_QINT32_REF=113]="DT_QINT32_REF",Jh[Jh.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",Jh[Jh.DT_QINT16_REF=115]="DT_QINT16_REF",Jh[Jh.DT_QUINT16_REF=116]="DT_QUINT16_REF",Jh[Jh.DT_UINT16_REF=117]="DT_UINT16_REF",Jh[Jh.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",Jh[Jh.DT_HALF_REF=119]="DT_HALF_REF",Jh[Jh.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",Jh[Jh.DT_VARIANT_REF=121]="DT_VARIANT_REF",Jh[Jh.DT_UINT32_REF=122]="DT_UINT32_REF",Jh[Jh.DT_UINT64_REF=123]="DT_UINT64_REF",Xh=Zh||(Zh={}),(Yh=Xh.CheckpointFormatVersion||(Xh.CheckpointFormatVersion={}))[Yh.LEGACY=0]="LEGACY",Yh[Yh.V1=1]="V1",Yh[Yh.V2=2]="V2";
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const Qh={};function em(e,t){const n={tfOpName:e,category:"custom",inputs:[],attrs:[],customExecutor:t};Qh[e]=n}function tm(e){return Qh[e]}function nm(e){delete Qh[e]}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rm(e,t,n,r,a){const s=t.inputParams[e];if(s&&void 0!==s.inputIndexStart){const e=s.inputIndexStart,o=0===s.inputIndexEnd?void 0:void 0===s.inputIndexEnd?e+1:s.inputIndexEnd,i=e<0?t.inputNames.length+e:e;if("tensor"===s.type)return am(t.inputNames[i],n,r,a);if("tensors"===s.type){const s=t.inputs.slice(e,o);return t.inputNames.slice(e,o).filter((e,t)=>{var n;return"NoOp"!==(null===(n=s[t])||void 0===n?void 0:n.op)}).map(e=>am(e,n,r,a))}const u=am(t.inputNames[i],n,r,a),l=u.dataSync();return"number"===s.type?l[0]:P(u.shape,l)}const o=t.attrParams[e];return o&&o.value}function am(e,t,n,r){const[a,s]=um(e,n);if(null!=r){const e=r.getHashTableHandleByName(a);if(null!=e)return e}const o=n.currentContextIds.find(e=>!!t[im(a,e)]);return void 0!==o?t[im(a,o)][s]:void 0}function sm(e,t,n){return t[im(e,n.currentContextId)]}function om(e,t){const[n,r,a]=um(e,t);return[im(n,t&&t.currentContextId),r,a]}function im(e,t){return t?`${e}-${t}`:e}function um(e,t){if(""===e)return["",0,void 0];const n=null!=t&&null!=t.parseNodeNameCache;if(n){const n=t.parseNodeNameCache.get(e);if(null!=n)return n}const r=e.split(":");let a;if(1===r.length)a=[e,0,void 0];else{const e=r[0],t=3===r.length?r[1]:void 0;a=[e,Number(r[r.length-1]),t]}return n&&t.parseNodeNameCache.set(e,a),a}function lm(e,t,n){let r=rm("pad",e,t,n);if("explicit"===r){r=rm("explicitPaddings",e,t,n);const a=[[0,0],[0,0],[0,0],[0,0]];for(let e=0;e<4;e++)a[e][0]=r[2*e],a[e][1]=r[2*e+1];return a}return r}function cm(e){return e.kept?e:Qo(e)}
/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}]},Symbol.toStringTag,{value:"Module"})),dm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}]},Symbol.toStringTag,{value:"Module"})),hm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}]},Symbol.toStringTag,{value:"Module"})),mm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}]},Symbol.toStringTag,{value:"Module"})),fm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}]},Symbol.toStringTag,{value:"Module"})),gm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}]},Symbol.toStringTag,{value:"Module"})),ym=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}]},Symbol.toStringTag,{value:"Module"})),bm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}]},Symbol.toStringTag,{value:"Module"})),wm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}]},Symbol.toStringTag,{value:"Module"})),Nm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}]},Symbol.toStringTag,{value:"Module"})),xm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}]},Symbol.toStringTag,{value:"Module"})),vm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}]},Symbol.toStringTag,{value:"Module"})),Tm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}]},Symbol.toStringTag,{value:"Module"})),Sm=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}]},Symbol.toStringTag,{value:"Module"})),km=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}]},Symbol.toStringTag,{value:"Module"})),Em=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}]},Symbol.toStringTag,{value:"Module"})),_m=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}]},Symbol.toStringTag,{value:"Module"})),Im=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}]},Symbol.toStringTag,{value:"Module"})),Am=Object.freeze(Object.defineProperty({__proto__:null,json:[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}]},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class Mm{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[].concat(...[pm,dm,hm,mm,fm,gm,ym,bm,wm,Nm,xm,vm,Tm,Sm,km,Em,_m,Im,Am].map(e=>e.json));this.opMappers=e.reduce((e,t)=>(e[t.tfOpName]=t,e),{})}transformGraph(e,t={}){const n=e.node,r=[],a=[],s=[],o=n.reduce((e,t)=>(e[t.name]=this.mapNode(t),t.op.startsWith("Placeholder")?r.push(e[t.name]):"Const"===t.op?a.push(e[t.name]):null!=t.input&&0!==t.input.length||s.push(e[t.name]),e),{});let i=[];const u=[];let l={},c={};null!=t&&(l=this.mapSignatureEntries(t.inputs),c=this.mapSignatureEntries(t.outputs));const p=Object.keys(o);p.forEach(e=>{const t=o[e];t.inputNames.forEach((e,n)=>{const[r,,a]=om(e),s=o[r];if(null!=s.outputs){const e=s.outputs.indexOf(a);if(-1!==e){const a=`${r}:${e}`;t.inputNames[n]=a}}t.inputs.push(s),s.children.push(t)})}),0===Object.keys(c).length?p.forEach(e=>{const t=o[e];0===t.children.length&&u.push(t)}):Object.keys(c).forEach(e=>{const[t]=om(e),n=o[t];null!=n&&(n.signatureKey=c[e],u.push(n))}),Object.keys(l).length>0?Object.keys(l).forEach(e=>{const[t]=om(e),n=o[t];n&&(n.signatureKey=l[e],i.push(n))}):i=r;let d={};null!=e.library&&null!=e.library.function&&(d=e.library.function.reduce((e,t)=>(e[t.signature.name]=this.mapFunction(t),e),{}));const h={nodes:o,inputs:i,outputs:u,weights:a,placeholders:r,signature:t,functions:d};return s.length>0&&(h.initNodes=s),h}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,n)=>(t[e[n].name]=n,t),{})}mapNode(e){const t=tm(e.op)||this.opMappers[e.op]||{};null==e.attr&&(e.attr={});const n={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(e=>e.startsWith("^")?e.slice(1):e),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return null!=t.inputs&&(n.inputParams=t.inputs.reduce((e,t)=>(e[t.name]={type:t.type,inputIndexStart:t.start,inputIndexEnd:t.end},e),{})),null!=t.attrs&&(n.attrParams=t.attrs.reduce((t,n)=>{const r=n.type;let a;switch(n.type){case"string":a=Om(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Om(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"string[]":a=Um(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Um(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"number":a=Cm(e.attr,n.tfName,n.defaultValue||0),void 0===a&&n.tfDeprecatedName&&(a=Cm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"number[]":a=Vm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Vm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"bool":a=Dm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Dm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"bool[]":a=Wm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Wm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"shape":a=Pm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Pm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"shape[]":a=jm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=jm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"dtype":a=Bm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Bm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"dtype[]":a=zm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=zm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"func":a=Fm(e.attr,n.tfName,n.defaultValue),void 0===a&&n.tfDeprecatedName&&(a=Fm(e.attr,n.tfDeprecatedName,n.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${n.type} for op: ${e.op}`)}return t[n.name]={value:a,type:r},t},{})),n}mapFunction(e){const t=e.nodeDef,n=[];let r={};null!=t&&(r=t.reduce((e,t)=>(e[t.name]=this.mapNode(t),"Const"===t.op&&n.push(e[t.name]),e),{}));const a=[],s=[];e.signature.inputArg.forEach(e=>{const[t]=om(e.name),n={name:t,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:Rm(e.type),type:"dtype"}},children:[]};n.signatureKey=e.name,a.push(n),r[t]=n});Object.keys(r).forEach(e=>{const t=r[e];t.inputNames.forEach((e,n)=>{const[a,,s]=om(e),o=r[a];if(null!=o.outputs){const e=o.outputs.indexOf(s);if(-1!==e){const r=`${a}:${e}`;t.inputNames[n]=r}}t.inputs.push(o),o.children.push(t)})});const o=e.ret;e.signature.outputArg.forEach(e=>{const[t,n]=om(o[e.name]),a=r[t];null!=a&&(a.defaultOutput=n,s.push(a))});const i=this.mapArgsToSignature(e);return{nodes:r,inputs:a,outputs:s,weights:n,placeholders:[],signature:i}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((e,t)=>(e[t.name]=this.mapArgToTensorInfo(t),e),{}),outputs:e.signature.outputArg.reduce((t,n)=>(t[n.name]=this.mapArgToTensorInfo(n,e.ret),t),{})}}mapArgToTensorInfo(e,t){let n=e.name;return null!=t&&(n=t[n]),{name:n,dtype:e.type}}}function $m(e,t){const n=Array.isArray(e)?String.fromCharCode.apply(null,e):function(e){const t=Y().global;if(void 0!==t.atob)return t.atob(e);if("undefined"!=typeof Buffer)return new Buffer(e,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}(e);return t?n:n.toLowerCase()}function Om(e,t,n,r=!1){const a=e[t];return null!=a?$m(a.s,r):n}function Dm(e,t,n){const r=e[t];return r?r.b:n}function Cm(e,t,n){const r=e[t]||{},a=null!=r.i?r.i:null!=r.f?r.f:n;return"number"==typeof a?a:parseInt(a,10)}function Rm(e){switch("string"==typeof e&&(e=Hh[e]),e){case Hh.DT_FLOAT:case Hh.DT_HALF:return"float32";case Hh.DT_INT32:case Hh.DT_INT64:case Hh.DT_INT8:case Hh.DT_UINT8:return"int32";case Hh.DT_BOOL:return"bool";case Hh.DT_DOUBLE:return"float32";case Hh.DT_STRING:return"string";case Hh.DT_COMPLEX64:case Hh.DT_COMPLEX128:return"complex64";default:return null}}function Fm(e,t,n){const r=e[t];return r&&r.func?r.func.name:n}function Bm(e,t,n){const r=e[t];return r&&r.type?Rm(r.type):n}function zm(e,t,n){const r=e[t];return r&&r.list&&r.list.type?r.list.type.map(e=>Rm(e)):n}function Lm(e){if(!e.unknownRank)return null!=e.dim?e.dim.map(e=>"number"==typeof e.size?e.size:parseInt(e.size,10)):[]}function Pm(e,t,n){const r=e[t];return r&&r.shape?Lm(r.shape):n}function Vm(e,t,n){const r=e[t];return r?((r.list.f&&r.list.f.length?r.list.f:r.list.i)||[]).map(e=>"number"==typeof e?e:parseInt(e,10)):n}function Um(e,t,n,r=!1){const a=e[t];return a&&a.list&&a.list.s?a.list.s.map(e=>$m(e,r)):n}function jm(e,t,n){const r=e[t];return r&&r.list&&r.list.shape?r.list.shape.map(e=>Lm(e)):n}function Wm(e,t,n){const r=e[t];return r&&r.list&&r.list.b?r.list.b:n}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class qm{constructor(e,t,n){this.node=e,this.tensorMap=t,this.context=n,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(e=>this.getInput(e)),null!=e.rawAttrs&&(this.attrs=Object.keys(e.rawAttrs).reduce((e,t)=>(e[t]=this.getAttr(t),e),{}))}getInput(e){return am(e,this.tensorMap,this.context)}getAttr(e,t){const n=this.node.rawAttrs[e];if(null!=n.tensor)return am(e,this.tensorMap,this.context);if(null!=n.i||null!=n.f)return Cm(this.node.rawAttrs,e,t);if(null!=n.s)return Om(this.node.rawAttrs,e,t);if(null!=n.b)return Dm(this.node.rawAttrs,e,t);if(null!=n.shape)return Pm(this.node.rawAttrs,e,t);if(null!=n.type)return Bm(this.node.rawAttrs,e,t);if(null!=n.list){if(null!=n.list.i||null!=n.list.f)return Vm(this.node.rawAttrs,e,t);if(null!=n.list.s)return Um(this.node.rawAttrs,e,t);if(null!=n.list.shape)return jm(this.node.rawAttrs,e,t);if(null!=n.list.b)return Wm(this.node.rawAttrs,e,t);if(null!=n.list.type)return zm(this.node.rawAttrs,e,t)}return t}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gm=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:_s,abs:si,acos:oi,acosh:ii,add:ti,addN:ui,all:li,any:ci,argMax:pi,argMin:di,asin:hi,asinh:mi,atan:fi,atan2:gi,atanh:yi,avgPool:Ci,avgPool3d:Ri,basicLSTMCell:Vi,batchNorm:ji,batchNorm2d:Wi,batchNorm3d:qi,batchNorm4d:Gi,batchToSpaceND:Ui,bincount:Ki,bitwiseAnd:Hi,booleanMaskAsync:qp,broadcastArgs:Ji,broadcastTo:Zi,buffer:Xo,cast:Yo,ceil:Xi,clipByValue:Qi,clone:Qo,complex:As,concat:Fi,concat1d:eu,concat2d:tu,concat3d:nu,concat4d:ru,conv1d:su,conv2d:au,conv2dTranspose:iu,conv3d:uu,conv3dTranspose:cu,cos:pu,cosh:du,cosineWindow:Qp,cumprod:hu,cumsum:mu,denseBincount:fu,depthToSpace:gu,depthwiseConv2d:yu,diag:bu,dilation2d:wu,div:ri,divNoNan:_u,dot:Iu,dropout:Xp,einsum:Au,elu:Mu,enclosingPowerOfTwo:Yp,ensureShape:$u,equal:Su,erf:Ou,euclideanNorm:Zu,exp:Xu,expandDims:Yu,expm1:Qu,eye:tl,fft:mp,fill:Yi,floor:nl,floorDiv:ni,fused:pd,gather:rl,gatherND:Zp,greater:al,greaterEqual:sl,ifft:fp,imag:ol,image:ah,inTopKAsync:ed,irfft:gp,isFinite:il,isInf:ul,isNaN:ll,leakyRelu:cl,less:pl,lessEqual:dl,linalg:sh,linspace:hl,localResponseNormalization:ml,log:fl,log1p:gl,logSigmoid:El,logSoftmax:Il,logSumExp:Al,logicalAnd:Ml,logicalNot:$l,logicalOr:Ol,logicalXor:Dl,losses:oh,lowerBound:Fl,matMul:Bi,max:Vu,maxPool:Bl,maxPool3d:zl,maxPoolWithArgmax:Ll,maximum:Pl,mean:Vl,meshgrid:Wl,min:Uu,minimum:ql,mirrorPad:Gl,mod:Kl,moments:Hl,movingAverage:Kp,mul:ai,multiRNNCell:Jl,multinomial:Zl,neg:Sl,norm:Ju,notEqual:Xl,oneHot:Yl,ones:jl,onesLike:Ql,op:Is,outerProduct:ec,pad:tc,pad1d:nc,pad2d:rc,pad3d:ac,pad4d:sc,pool:ic,pow:ju,prelu:uc,print:ei,prod:lc,raggedGather:cc,raggedRange:pc,raggedTensorToTensor:dc,rand:hc,randomGamma:Pc,randomNormal:Vc,randomStandardNormal:Uc,randomUniform:jc,randomUniformInt:Wc,range:qc,real:Gc,reciprocal:Kc,relu:Hc,relu6:Jc,reshape:Di,reverse:Zc,reverse1d:Xc,reverse2d:Yc,reverse3d:Qc,reverse4d:ep,rfft:bp,round:tp,rsqrt:np,scalar:Wu,scatterND:Hp,searchSorted:Rl,selu:rp,separableConv2d:ap,setdiff1dAsync:sp,sigmoid:zi,sign:op,signal:rh,sin:ip,sinh:up,slice:Li,slice1d:lp,slice2d:cp,slice3d:pp,slice4d:dp,softmax:hp,softplus:kl,spaceToBatchND:oc,sparse:ih,sparseToDense:Jp,spectral:nh,split:yp,sqrt:qu,square:Gu,squaredDifference:wp,squeeze:Np,stack:xp,step:vp,stridedSlice:Tp,string:uh,sub:_l,sum:Ku,tan:Sp,tanh:Pi,tensor:$s,tensor1d:kp,tensor2d:Ep,tensor3d:_p,tensor4d:Ip,tensor5d:Ap,tensor6d:Mp,tensorScatterUpdate:Rp,tile:el,topk:Fp,transpose:Gp,truncatedNormal:Bp,unique:zp,unsortedSegmentSum:Lp,unstack:Pp,upperBound:Vp,variable:Up,where:ku,whereAsync:Wp,zeros:Ul,zerosLike:Eu},Symbol.toStringTag,{value:"Module"}));
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function Km(e,t,n=""){if("number"!=typeof e&&"number"!=typeof t){d(e.length===t.length,()=>n+` Shapes ${e} and ${t} must match`);for(let r=0;r<e.length;r++){const a=e[r],s=t[r];d(a<0||s<0||a===s,()=>n+` Shapes ${e} and ${t} must match`)}}}function Hm(e){return"number"!=typeof e&&!e.some(e=>e<0)}function Jm(e,t,n){let r=Zm(e,n);const a=!Hm(r);if(a&&0===t.length)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${r}`);if(a&&t.forEach(e=>{r=Zm(e.shape,r)}),!Hm(r))throw new Error(`Non-fully-defined elementShape: ${r}`);return r}function Zm(e,t){if("number"==typeof e)return t;if("number"==typeof t)return e;if(e.length!==t.length)throw new Error(`Incompatible ranks during merge: ${e} vs. ${t}`);const n=[];for(let r=0;r<e.length;++r){const a=e[r],s=t[r];if(a>=0&&s>=0&&a!==s)throw new Error(`Incompatible shape during merge: ${e} vs. ${t}`);n[r]=a>=0?a:s}return n}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Xm{constructor(e,t,n,r,a,s,o){this.name=e,this.dtype=t,this.maxSize=n,this.elementShape=r,this.identicalElementShapes=a,this.dynamicSize=s,this.clearAfterRead=o,this.tensors=[],this.closed_=!1,this.idTensor=Wu(0),Ws(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{null!=e&&e.has(t.tensor.id)||t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const t=this.tensors[e];if(t.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(e=>this.read(e))}write(e,t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const n=this.tensors[e]||{};if(t.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},\n          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(0!==this.size()||null!=this.elementShape&&0!==this.elementShape.length||(this.elementShape=t.shape),Km(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),n.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(n.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);n.tensor=t,Ws(t),n.written=!0,this.tensors[e]=n}writeMany(e,t){if(e.length!==t.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((e,n)=>this.write(e,t[n]))}gather(e,t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let t=0;t<this.size();t++)e.push(t)}if(0===e.length)return $s([],[0].concat(this.elementShape));const n=this.readMany(e);return Km(this.elementShape,n[0].shape,"TensorArray shape mismatch: "),xp(n,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(0===this.size())return $s([],[0].concat(this.elementShape));const t=[];for(let r=0;r<this.size();r++)t.push(r);const n=this.readMany(t);return Km(this.elementShape,n[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${n[0].shape})`),Fi(n,0)}scatter(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const n=Math.max(...e);if(!this.dynamicSize&&n>=this.maxSize)throw new Error(`Max index must be < array size (${n}  vs. ${this.maxSize})`);this.writeMany(e,Pp(t,0))}split(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let n=0;const r=e.map(e=>(n+=e,n));if(n!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to\n          tensor.shape[0], but sum of lengths is\n        ${n}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const a=0===n?0:t.size/n,s=[];Us(()=>{t=Di(t,[1,n,a]);for(let n=0;n<e.length;++n){const o=[0,0===n?0:r[n-1],0],i=[1,e[n],a];s[n]=Di(Li(t,o,i),this.elementShape)}return s});const o=[];for(let i=0;i<e.length;i++)o[i]=i;this.writeMany(o,s)}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ym{get id(){return this.idTensor.id}constructor(e,t,n,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=n,null!=e&&e.forEach(e=>{if(n!==e.dtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${e.dtype}`);Km(t,e.shape,"TensorList shape mismatch: "),Ws(e)}),this.idTensor=Wu(0),this.maxNumElements=r,Ws(this.idTensor)}copy(){return new Ym([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{null!=e&&e.has(t.id)||t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,n=-1){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(-1!==n&&this.tensors.length!==n)throw new Error(`Operation expected a list with ${n} elements but got a list with ${this.tensors.length} elements.`);Km(e,this.elementShape,"TensorList shape mismatch: ");const r=Jm(this.elementShape,this.tensors,e);return Us(()=>{const e=this.tensors.map(e=>Di(e,r));return xp(e,0)})}popBack(e,t){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(0===this.size())throw new Error("Trying to pop from an empty list.");const n=Jm(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,Km(r.shape,e,"TensorList shape mismatch: "),Di(r,n)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(Km(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");Ws(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(-1!==this.maxNumElements&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const t=new Ym([],this.elementShape,this.elementDtype,this.maxNumElements);t.tensors.length=e;for(let n=0;n<Math.min(this.tensors.length,e);++n)t.tensors[n]=this.tensors[n];return t}getItem(e,t,n){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(null==this.tensors[e])throw new Error(`element at index ${e} is null.`);Km(this.tensors[e].shape,t,"TensorList shape mismatch: ");const r=Jm(this.elementShape,this.tensors,t);return Di(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||-1!==this.maxNumElements&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);Km(this.elementShape,t.shape,"TensorList shape mismatch: "),Ws(t),null!=this.tensors[e]&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,n){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);Km(this.elementShape,n,"TensorList shape mismatch: "),e=e.slice(0,this.size());const r=Jm(this.elementShape,this.tensors,n);return 0===e.length?$s([],[0].concat(r)):Us(()=>{const t=e.map(e=>Di(this.tensors[e],r));return xp(t,0)})}concat(e,t){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);Km(this.elementShape,t,"TensorList shape mismatch: ");const n=Jm(this.elementShape,this.tensors,t);return 0===this.size()?$s([],[0].concat(n)):Us(()=>{const e=this.tensors.map(e=>Di(e,n));return Fi(e,0)})}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const Qm=async(e,t,n)=>{switch(e.op){case"If":case"StatelessIf":{const r=rm("thenBranch",e,t,n),a=rm("elseBranch",e,t,n),s=rm("cond",e,t,n),o=rm("args",e,t,n);return(await s.data())[0]?n.functionMap[r].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap):n.functionMap[a].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap)}case"While":case"StatelessWhile":{const r=rm("body",e,t,n),a=rm("cond",e,t,n),s=rm("args",e,t,n),o=await n.functionMap[a].executeFunctionAsync(s,n.tensorArrayMap,n.tensorListMap),i=s.map(e=>e.id);let u=await o[0].data();o.forEach(e=>{e.kept||-1!==i.indexOf(e.id)||e.dispose()});let l=s;for(;u[0];){const e=l;l=await n.functionMap[r].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);const t=l.map(e=>e.id);e.forEach(e=>{e.kept||-1!==i.indexOf(e.id)||-1!==t.indexOf(e.id)||e.dispose()});const s=await n.functionMap[a].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);u=await s[0].data(),s.forEach(e=>{e.kept||-1!==i.indexOf(e.id)||-1!==t.indexOf(e.id)||e.dispose()})}return l}case"LoopCond":return[cm(rm("pred",e,t,n))];case"Switch":{const r=rm("pred",e,t,n);let a=rm("data",e,t,n);return a.kept||(a=cm(a)),(await r.data())[0]?[void 0,a]:[a,void 0]}case"Merge":{const r=e.inputNames.find(e=>void 0!==am(e,t,n));if(r){return[cm(am(r,t,n))]}return}case"Enter":{const r=rm("frameName",e,t,n),a=rm("tensor",e,t,n);return n.enterFrame(r),[cm(a)]}case"Exit":{const r=rm("tensor",e,t,n);return n.exitFrame(),[cm(r)]}case"NextIteration":{const r=rm("tensor",e,t,n);return n.nextIteration(),[cm(r)]}case"TensorArrayV3":{const r=rm("size",e,t,n),a=rm("dtype",e,t,n),s=rm("elementShape",e,t,n),o=rm("dynamicSize",e,t,n),i=rm("clearAfterRead",e,t,n),u=rm("identicalElementShapes",e,t,n),l=rm("name",e,t,n),c=new Xm(l,a,r,s,u,o,i);return n.addTensorArray(c),[c.idTensor,Wu(1)]}case"TensorArrayWriteV3":{const r=rm("tensorArrayId",e,t,n),a=rm("index",e,t,n),s=rm("tensor",e,t,n),o=n.getTensorArray(r.id);return o.write(a,s),[o.idTensor]}case"TensorArrayReadV3":{const r=rm("tensorArrayId",e,t,n),a=rm("index",e,t,n);return[n.getTensorArray(r.id).read(a)]}case"TensorArrayGatherV3":{const r=rm("tensorArrayId",e,t,n),a=rm("indices",e,t,n),s=rm("dtype",e,t,n);return[n.getTensorArray(r.id).gather(a,s)]}case"TensorArrayScatterV3":{const r=rm("tensorArrayId",e,t,n),a=rm("indices",e,t,n),s=rm("tensor",e,t,n),o=n.getTensorArray(r.id);return o.scatter(a,s),[o.idTensor]}case"TensorArrayConcatV3":{const r=rm("tensorArrayId",e,t,n),a=n.getTensorArray(r.id),s=rm("dtype",e,t,n);return[a.concat(s)]}case"TensorArraySplitV3":{const r=rm("tensorArrayId",e,t,n),a=rm("tensor",e,t,n),s=rm("lengths",e,t,n),o=n.getTensorArray(r.id);return o.split(s,a),[o.idTensor]}case"TensorArraySizeV3":{const r=rm("tensorArrayId",e,t,n);return[Wu(n.getTensorArray(r.id).size(),"int32")]}case"TensorArrayCloseV3":{const r=rm("tensorArrayId",e,t,n),a=n.getTensorArray(r.id);return a.clearAndClose(),[a.idTensor]}case"TensorListSetItem":{const r=rm("tensorListId",e,t,n),a=rm("index",e,t,n),s=rm("tensor",e,t,n),o=n.getTensorList(r.id);return o.setItem(a,s),[o.idTensor]}case"TensorListGetItem":{const r=rm("tensorListId",e,t,n),a=rm("index",e,t,n),s=rm("elementShape",e,t,n),o=rm("elementDType",e,t,n);return[n.getTensorList(r.id).getItem(a,s,o)]}case"TensorListScatterV2":case"TensorListScatter":{const r=rm("indices",e,t,n),a=function(e,t,n,r){if(t.length!==e.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${e.shape[0]}`);const a=Math.max(...t);if(null!=r&&-1!==r&&a>=r)throw new Error(`Max index must be < array size (${a}  vs. ${r})`);const s=new Ym([],n,e.dtype,r),o=Pp(e,0);return t.forEach((e,t)=>{s.setItem(e,o[t])}),s}(rm("tensor",e,t,n),r,rm("elementShape",e,t,n),rm("numElements",e,t,n));return n.addTensorList(a),[a.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const r=rm("elementShape",e,t,n),a=rm("elementDType",e,t,n);let s;s="TensorListReserve"===e.op?"numElements":"maxNumElements";const o=rm(s,e,t,n),i=function(e,t,n,r){return new Ym([],e,t,r)}(r,a,0,"TensorListReserve"===e.op?-1:o);return n.addTensorList(i),[i.idTensor]}case"TensorListGather":{const r=rm("tensorListId",e,t,n),a=rm("indices",e,t,n),s=rm("elementShape",e,t,n),o=rm("elementDType",e,t,n);return[n.getTensorList(r.id).gather(a,o,s)]}case"TensorListStack":{const r=rm("tensorListId",e,t,n),a=rm("elementShape",e,t,n),s=rm("elementDType",e,t,n),o=rm("numElements",e,t,n);return[n.getTensorList(r.id).stack(a,s,o)]}case"TensorListFromTensor":{const r=function(e,t,n){const r=e.dtype;if(e.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${e.shape}`);if(e.dtype!==n)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${n}`);Km(e.shape.slice(1),t,"TensorList shape mismatch: ");const a=Pp(e);return new Ym(a,t,r)}(rm("tensor",e,t,n),rm("elementShape",e,t,n),rm("elementDType",e,t,n));return n.addTensorList(r),[r.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const r=rm("tensorListId",e,t,n),a=n.getTensorList(r.id),s=rm("dtype",e,t,n),o=rm("elementShape",e,t,n);return[a.concat(s,o)]}case"TensorListPushBack":{const r=rm("tensorListId",e,t,n),a=rm("tensor",e,t,n),s=n.getTensorList(r.id);return s.pushBack(a),[s.idTensor]}case"TensorListPopBack":{const r=rm("tensorListId",e,t,n),a=rm("elementShape",e,t,n),s=rm("elementDType",e,t,n);return[n.getTensorList(r.id).popBack(a,s)]}case"TensorListSplit":{const r=rm("tensor",e,t,n),a=rm("elementShape",e,t,n),s=function(e,t,n){let r=0;const a=t.map(e=>(r+=e,r));if(r!==e.shape[0])throw new Error(`Expected sum of lengths to be equal to\n          tensor.shape[0], but sum of lengths is\n        ${r}, and tensor's shape is: ${e.shape}`);const s=Zm(e.shape.slice(1),n),o=0===r?0:e.size/r,i=Us(()=>{const n=[];e=Di(e,[1,r,o]);for(let r=0;r<t.length;++r){const i=[0,0===r?0:a[r-1],0],u=[1,t[r],o];n[r]=Di(Li(e,i,u),s)}return e.dispose(),n}),u=new Ym([],n,e.dtype,t.length);for(let l=0;l<i.length;l++)u.setItem(l,i[l]);return u}(r,rm("lengths",e,t,n),a);return n.addTensorList(s),[s.idTensor]}case"TensorListLength":{const r=rm("tensorListId",e,t,n);return[Wu(n.getTensorList(r.id).size(),"int32")]}case"TensorListResize":{const r=rm("tensorListId",e,t,n),a=rm("size",e,t,n),s=n.getTensorList(r.id).resize(a);return n.addTensorList(s),[s.idTensor]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ef(e,t,n){const[r,a]=rm("fusedOps",e,t,n),s="biasadd"===r,o=!s,i="prelu"===a,u="fusedbatchnorm"===r,l=rm("numArgs",e,t,n);if(s){if(i&&2!==l)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&s&&1!==l)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(u)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const c=rm("strides",e,t,n),p=lm(e,t,n),d=rm("dataFormat",e,t,n).toUpperCase(),h=rm("dilations",e,t,n);let[m,f]=rm("args",e,t,n);o&&(f=m,m=void 0);return{stride:c,pad:p,dataFormat:d,dilations:h,biasArg:m,preluArg:f,activationFunc:a,leakyreluAlpha:rm("leakyreluAlpha",e,t,n)}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function tf(e,t,n){return{boxes:rm("boxes",e,t,n),scores:rm("scores",e,t,n),maxOutputSize:rm("maxOutputSize",e,t,n),iouThreshold:rm("iouThreshold",e,t,n),scoreThreshold:rm("scoreThreshold",e,t,n),softNmsSigma:rm("softNmsSigma",e,t,n)}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
class nf{get id(){return this.handle.id}constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=Wu(0),this.tensorMap=new Map,Ws(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return Wu(this.size(),"int32")}async import(e,t){this.checkKeyAndValueTensor(e,t);const n=await e.data();return this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),Us(()=>{const e=Pp(t),r=n.length,a=e.length;d(r===a,()=>`The number of elements doesn't match, keys has ${r} elements, the values has ${a} elements.`);for(let t=0;t<r;t++){const r=n[t],a=e[t];Ws(a),this.tensorMap.set(r,a)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);const n=await e.data();return Us(()=>{const e=[];for(let r=0;r<n.length;r++){const a=n[r],s=this.findWithDefault(a,t);e.push(s)}return xp(e)})}findWithDefault(e,t){const n=this.tensorMap.get(e);return null!=n?n:t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}}
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function rf(e,t,n,r,a=Us){const s=((e,t,n)=>{switch(e.category){case"arithmetic":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"BiasAdd":case"AddV2":case"Add":return[r.add(rm("a",e,t,n),rm("b",e,t,n))];case"AddN":return[r.addN(rm("tensors",e,t,n))];case"FloorMod":case"Mod":return[r.mod(rm("a",e,t,n),rm("b",e,t,n))];case"Mul":return[r.mul(rm("a",e,t,n),rm("b",e,t,n))];case"RealDiv":case"Div":return[r.div(rm("a",e,t,n),rm("b",e,t,n))];case"DivNoNan":return[r.divNoNan(rm("a",e,t,n),rm("b",e,t,n))];case"FloorDiv":return[r.floorDiv(rm("a",e,t,n),rm("b",e,t,n))];case"Sub":return[r.sub(rm("a",e,t,n),rm("b",e,t,n))];case"Minimum":return[r.minimum(rm("a",e,t,n),rm("b",e,t,n))];case"Maximum":return[r.maximum(rm("a",e,t,n),rm("b",e,t,n))];case"Pow":return[r.pow(rm("a",e,t,n),rm("b",e,t,n))];case"SquaredDifference":return[r.squaredDifference(rm("a",e,t,n),rm("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"basic_math":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"Abs":case"ComplexAbs":return[r.abs(rm("x",e,t,n))];case"Acos":return[r.acos(rm("x",e,t,n))];case"Acosh":return[r.acosh(rm("x",e,t,n))];case"Asin":return[r.asin(rm("x",e,t,n))];case"Asinh":return[r.asinh(rm("x",e,t,n))];case"Atan":return[r.atan(rm("x",e,t,n))];case"Atan2":return[r.atan2(rm("x",e,t,n),rm("y",e,t,n))];case"Atanh":return[r.atanh(rm("x",e,t,n))];case"Ceil":return[r.ceil(rm("x",e,t,n))];case"Complex":return[r.complex(rm("real",e,t,n),rm("imag",e,t,n))];case"Cos":return[r.cos(rm("x",e,t,n))];case"Cosh":return[r.cosh(rm("x",e,t,n))];case"Elu":return[r.elu(rm("x",e,t,n))];case"Erf":return[r.erf(rm("x",e,t,n))];case"Exp":return[r.exp(rm("x",e,t,n))];case"Expm1":return[r.expm1(rm("x",e,t,n))];case"Floor":return[r.floor(rm("x",e,t,n))];case"Log":return[r.log(rm("x",e,t,n))];case"Log1p":return[r.log1p(rm("x",e,t,n))];case"Imag":return[r.imag(rm("x",e,t,n))];case"Neg":return[r.neg(rm("x",e,t,n))];case"Reciprocal":return[r.reciprocal(rm("x",e,t,n))];case"Real":return[r.real(rm("x",e,t,n))];case"Relu":return[r.relu(rm("x",e,t,n))];case"Round":return[r.round(rm("x",e,t,n))];case"Selu":return[r.selu(rm("x",e,t,n))];case"Sigmoid":return[r.sigmoid(rm("x",e,t,n))];case"Sin":return[r.sin(rm("x",e,t,n))];case"Sign":return[r.sign(rm("x",e,t,n))];case"Sinh":return[r.sinh(rm("x",e,t,n))];case"Softplus":return[r.softplus(rm("x",e,t,n))];case"Sqrt":return[r.sqrt(rm("x",e,t,n))];case"Square":return[r.square(rm("x",e,t,n))];case"Tanh":return[r.tanh(rm("x",e,t,n))];case"Tan":return[r.tan(rm("x",e,t,n))];case"ClipByValue":return[r.clipByValue(rm("x",e,t,n),rm("clipValueMin",e,t,n),rm("clipValueMax",e,t,n))];case"Relu6":return[r.relu6(rm("x",e,t,n))];case"Rsqrt":return[r.rsqrt(am(e.inputNames[0],t,n))];case"LeakyRelu":return[r.leakyRelu(rm("x",e,t,n),rm("alpha",e,t,n))];case"Prelu":return[r.prelu(rm("x",e,t,n),rm("alpha",e,t,n))];case"IsNan":return[r.isNaN(am(e.inputNames[0],t,n))];case"IsInf":return[r.isInf(am(e.inputNames[0],t,n))];case"IsFinite":return[r.isFinite(am(e.inputNames[0],t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"control":return Qm(e,t,n);case"convolution":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"Conv1D":{const a=rm("stride",e,t,n),s=rm("pad",e,t,n),o=rm("dataFormat",e,t,n).toUpperCase(),i=rm("dilation",e,t,n);return[r.conv1d(rm("x",e,t,n),rm("filter",e,t,n),a,s,o,i)]}case"Conv2D":{const a=rm("strides",e,t,n),s=lm(e,t,n),o=rm("dataFormat",e,t,n).toUpperCase(),i=rm("dilations",e,t,n);return[r.conv2d(rm("x",e,t,n),rm("filter",e,t,n),[a[1],a[2]],s,o,[i[1],i[2]])]}case"_FusedConv2D":{const{stride:a,pad:s,dataFormat:o,dilations:i,biasArg:u,preluArg:l,activationFunc:c,leakyreluAlpha:p}=ef(e,t,n);return[r.fused.conv2d({x:rm("x",e,t,n),filter:rm("filter",e,t,n),strides:[a[1],a[2]],pad:s,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:c,preluActivationWeights:l,leakyreluAlpha:p})]}case"FusedDepthwiseConv2dNative":{const{stride:a,pad:s,dataFormat:o,dilations:i,biasArg:u,preluArg:l,activationFunc:c,leakyreluAlpha:p}=ef(e,t,n);return[r.fused.depthwiseConv2d({x:rm("x",e,t,n),filter:rm("filter",e,t,n),strides:[a[1],a[2]],pad:s,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:c,preluActivationWeights:l,leakyreluAlpha:p})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const a=rm("outputShape",e,t,n),s=rm("strides",e,t,n),o=lm(e,t,n);return[r.conv2dTranspose(rm("x",e,t,n),rm("filter",e,t,n),a,[s[1],s[2]],o)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const a=rm("strides",e,t,n),s=lm(e,t,n),o=rm("dilations",e,t,n),i=rm("dataFormat",e,t,n).toUpperCase();return[r.depthwiseConv2d(rm("input",e,t,n),rm("filter",e,t,n),[a[1],a[2]],s,i,[o[1],o[2]])]}case"Conv3D":{const a=rm("strides",e,t,n),s=rm("pad",e,t,n),o=rm("dataFormat",e,t,n).toUpperCase(),i=rm("dilations",e,t,n);return[r.conv3d(rm("x",e,t,n),rm("filter",e,t,n),[a[1],a[2],a[3]],s,o,[i[1],i[2],i[3]])]}case"AvgPool":{const a=rm("strides",e,t,n),s=rm("pad",e,t,n),o=rm("kernelSize",e,t,n);return[r.avgPool(rm("x",e,t,n),[o[1],o[2]],[a[1],a[2]],s)]}case"MaxPool":{const a=rm("strides",e,t,n),s=rm("pad",e,t,n),o=rm("kernelSize",e,t,n);return[r.maxPool(rm("x",e,t,n),[o[1],o[2]],[a[1],a[2]],s)]}case"MaxPoolWithArgmax":{const a=rm("strides",e,t,n),s=rm("pad",e,t,n),o=rm("kernelSize",e,t,n),i=rm("includeBatchInIndex",e,t,n),{result:u,indexes:l}=r.maxPoolWithArgmax(rm("x",e,t,n),[o[1],o[2]],[a[1],a[2]],s,i);return[u,l]}case"AvgPool3D":{const a=rm("strides",e,t,n),s=rm("pad",e,t,n),o=rm("kernelSize",e,t,n);return[r.avgPool3d(rm("x",e,t,n),[o[1],o[2],o[3]],[a[1],a[2],a[3]],s)]}case"MaxPool3D":{const a=rm("strides",e,t,n),s=rm("pad",e,t,n),o=rm("kernelSize",e,t,n);return[r.maxPool3d(rm("x",e,t,n),[o[1],o[2],o[3]],[a[1],a[2],a[3]],s)]}case"Dilation2D":{const a=rm("strides",e,t,n),s=rm("pad",e,t,n),o=rm("dilations",e,t,n),i=a[1],u=a[2],l=o[1],c=o[2];return[r.dilation2d(rm("x",e,t,n),rm("filter",e,t,n),[i,u],s,[l,c],"NHWC")]}default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"creation":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"Fill":{const a=rm("shape",e,t,n),s=rm("dtype",e,t,n),o=rm("value",e,t,n);return[r.fill(a,o,s)]}case"LinSpace":{const a=rm("start",e,t,n),s=rm("stop",e,t,n),o=rm("num",e,t,n);return[r.linspace(a,s,o)]}case"Multinomial":{const a=rm("logits",e,t,n),s=rm("numSamples",e,t,n),o=rm("seed",e,t,n);return[r.multinomial(a,s,o)]}case"OneHot":{const a=rm("indices",e,t,n),s=rm("depth",e,t,n),o=rm("onValue",e,t,n),i=rm("offValue",e,t,n),u=rm("dtype",e,t,n);return[r.oneHot(a,s,o,i,u)]}case"Ones":return[r.ones(rm("shape",e,t,n),rm("dtype",e,t,n))];case"OnesLike":return[r.onesLike(rm("x",e,t,n))];case"RandomStandardNormal":return[r.randomStandardNormal(rm("shape",e,t,n),rm("dtype",e,t,n),rm("seed",e,t,n))];case"RandomUniform":return[r.randomUniform(rm("shape",e,t,n),rm("minval",e,t,n),rm("maxval",e,t,n),rm("dtype",e,t,n))];case"RandomUniformInt":return[r.randomUniformInt(rm("shape",e,t,n),rm("minval",e,t,n),rm("maxval",e,t,n),rm("seed",e,t,n))];case"Range":{const a=rm("start",e,t,n),s=rm("stop",e,t,n),o=rm("step",e,t,n);return[r.range(a,s,o,rm("dtype",e,t,n))]}case"TruncatedNormal":{const a=rm("shape",e,t,n),s=rm("mean",e,t,n),o=rm("stdDev",e,t,n),i=rm("seed",e,t,n);return[r.truncatedNormal(a,s,o,rm("dtype",e,t,n),i)]}case"Zeros":return[r.zeros(rm("shape",e,t,n),rm("dtype",e,t,n))];case"ZerosLike":return[r.zerosLike(rm("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"dynamic":return(async(e,t,n,r,a=Gm)=>{switch(e.op){case"NonMaxSuppressionV5":{const{boxes:r,scores:s,maxOutputSize:o,iouThreshold:i,scoreThreshold:u,softNmsSigma:l}=tf(e,t,n),c=await a.image.nonMaxSuppressionWithScoreAsync(r,s,o,i,u,l);return[c.selectedIndices,c.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:r,scores:s,maxOutputSize:o,iouThreshold:i,scoreThreshold:u}=tf(e,t,n),l=rm("padToMaxOutputSize",e,t,n),c=await a.image.nonMaxSuppressionPaddedAsync(r,s,o,i,u,l);return[c.selectedIndices,c.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:r,scores:s,maxOutputSize:o,iouThreshold:i,scoreThreshold:u}=tf(e,t,n);return[await a.image.nonMaxSuppressionAsync(r,s,o,i,u)]}case"Where":{const r=a.cast(rm("condition",e,t,n),"bool"),s=[await a.whereAsync(r)];return r.dispose(),s}case"ListDiff":return a.setdiff1dAsync(rm("x",e,t,n),rm("y",e,t,n));default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n);case"evaluation":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"LowerBound":{const a=rm("sortedSequence",e,t,n),s=rm("values",e,t,n);return[r.lowerBound(a,s)]}case"TopKV2":{const a=rm("x",e,t,n),s=rm("k",e,t,n),o=rm("sorted",e,t,n),i=r.topk(a,s,o);return[i.values,i.indices]}case"UpperBound":{const a=rm("sortedSequence",e,t,n),s=rm("values",e,t,n);return[r.upperBound(a,s)]}case"Unique":{const a=rm("x",e,t,n),s=r.unique(a);return[s.values,s.indices]}case"UniqueV2":{const a=rm("x",e,t,n),s=rm("axis",e,t,n),o=r.unique(a,s);return[o.values,o.indices]}default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"image":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"ResizeBilinear":{const a=rm("images",e,t,n),s=rm("size",e,t,n),o=rm("alignCorners",e,t,n),i=rm("halfPixelCenters",e,t,n);return[r.image.resizeBilinear(a,[s[0],s[1]],o,i)]}case"ResizeNearestNeighbor":{const a=rm("images",e,t,n),s=rm("size",e,t,n),o=rm("alignCorners",e,t,n),i=rm("halfPixelCenters",e,t,n);return[r.image.resizeNearestNeighbor(a,[s[0],s[1]],o,i)]}case"CropAndResize":{const a=rm("image",e,t,n),s=rm("boxes",e,t,n),o=rm("boxInd",e,t,n),i=rm("cropSize",e,t,n),u=rm("method",e,t,n),l=rm("extrapolationValue",e,t,n);return[r.image.cropAndResize(a,s,o,i,u,l)]}case"ImageProjectiveTransformV3":{const a=rm("images",e,t,n),s=rm("transforms",e,t,n),o=rm("outputShape",e,t,n),i=rm("fillValue",e,t,n),u=rm("interpolation",e,t,n),l=rm("fillMode",e,t,n);return[r.image.transform(a,s,u.toLowerCase(),l.toLowerCase(),i,o)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"graph":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"Const":return t[e.name];case"PlaceholderWithDefault":const a=rm("default",e,t,n);return[am(e.name,t,n)||a];case"Placeholder":return[am(e.name,t,n)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":case"Snapshot":return[cm(rm("x",e,t,n))];case"IdentityN":return rm("x",e,t,n).map(e=>cm(e));case"Shape":return[r.tensor1d(rm("x",e,t,n).shape,"int32")];case"ShapeN":return rm("x",e,t,n).map(e=>r.tensor1d(e.shape));case"Size":return[r.scalar(rm("x",e,t,n).size,"int32")];case"Rank":return[r.scalar(rm("x",e,t,n).rank,"int32")];case"NoOp":return[r.scalar(1)];case"Print":const s=rm("x",e,t,n),o=rm("data",e,t,n);rm("message",e,t,n),rm("summarize",e,t,n);for(let e=0;e<o.length;e++);return[s];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"logical":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"Equal":return[r.equal(rm("a",e,t,n),rm("b",e,t,n))];case"NotEqual":return[r.notEqual(rm("a",e,t,n),rm("b",e,t,n))];case"Greater":return[r.greater(rm("a",e,t,n),rm("b",e,t,n))];case"GreaterEqual":return[r.greaterEqual(rm("a",e,t,n),rm("b",e,t,n))];case"Less":return[r.less(rm("a",e,t,n),rm("b",e,t,n))];case"LessEqual":return[r.lessEqual(rm("a",e,t,n),rm("b",e,t,n))];case"LogicalAnd":return[r.logicalAnd(rm("a",e,t,n),rm("b",e,t,n))];case"LogicalNot":return[r.logicalNot(rm("a",e,t,n))];case"LogicalOr":return[r.logicalOr(rm("a",e,t,n),rm("b",e,t,n))];case"Select":case"SelectV2":return[r.where(rm("condition",e,t,n),rm("a",e,t,n),rm("b",e,t,n))];case"BitwiseAnd":return[r.bitwiseAnd(rm("a",e,t,n),rm("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"matrices":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[r.matMul(rm("a",e,t,n),rm("b",e,t,n),rm("transposeA",e,t,n),rm("transposeB",e,t,n))];case"Einsum":return[r.einsum(rm("equation",e,t,n),...rm("tensors",e,t,n))];case"Transpose":return[r.transpose(rm("x",e,t,n),rm("perm",e,t,n))];case"_FusedMatMul":const[a,s]=rm("fusedOps",e,t,n),o="biasadd"===a,i="prelu"===s,u=rm("numArgs",e,t,n),l=rm("leakyreluAlpha",e,t,n);if(o){if(i&&2!==u)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&1!==u)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[c,p]=rm("args",e,t,n);return[r.fused.matMul({a:rm("a",e,t,n),b:rm("b",e,t,n),transposeA:rm("transposeA",e,t,n),transposeB:rm("transposeB",e,t,n),bias:c,activation:s,preluActivationWeights:p,leakyreluAlpha:l})];case"MatrixBandPart":return[r.linalg.bandPart(rm("a",e,t,n),rm("numLower",e,t,n),rm("numUpper",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"normalization":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"EuclideanNorm":return[r.euclideanNorm(rm("x",e,t,n),rm("axis",e,t,n),rm("keepDims",e,t,n))];case"FusedBatchNorm":case"FusedBatchNormV2":case"FusedBatchNormV3":return[r.batchNorm(rm("x",e,t,n),rm("mean",e,t,n),rm("variance",e,t,n),rm("offset",e,t,n),rm("scale",e,t,n),rm("epsilon",e,t,n))];case"LRN":return[r.localResponseNormalization(rm("x",e,t,n),rm("radius",e,t,n),rm("bias",e,t,n),rm("alpha",e,t,n),rm("beta",e,t,n))];case"Softmax":return[r.softmax(rm("x",e,t,n))];case"LogSoftmax":return[r.logSoftmax(rm("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"ragged":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"RaggedGather":{const{outputNestedSplits:a,outputDenseValues:s}=r.raggedGather(rm("paramsNestedSplits",e,t,n),rm("paramsDenseValues",e,t,n),rm("indices",e,t,n),rm("outputRaggedRank",e,t,n));return a.concat(s)}case"RaggedRange":{const{rtNestedSplits:a,rtDenseValues:s}=r.raggedRange(rm("starts",e,t,n),rm("limits",e,t,n),rm("splits",e,t,n));return[a,s]}case"RaggedTensorToTensor":return[r.raggedTensorToTensor(rm("shape",e,t,n),rm("values",e,t,n),rm("defaultValue",e,t,n),rm("rowPartitionTensors",e,t,n),rm("rowPartitionTypes",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"reduction":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"Max":{const a=rm("axis",e,t,n),s=rm("keepDims",e,t,n);return[r.max(rm("x",e,t,n),a,s)]}case"Mean":{const a=rm("axis",e,t,n),s=rm("keepDims",e,t,n);return[r.mean(rm("x",e,t,n),a,s)]}case"Min":{const a=rm("axis",e,t,n),s=rm("keepDims",e,t,n);return[r.min(rm("x",e,t,n),a,s)]}case"Sum":{const a=rm("axis",e,t,n),s=rm("keepDims",e,t,n);return[r.sum(rm("x",e,t,n),a,s)]}case"All":{const a=rm("axis",e,t,n),s=rm("keepDims",e,t,n);return[r.all(rm("x",e,t,n),a,s)]}case"Any":{const a=rm("axis",e,t,n),s=rm("keepDims",e,t,n);return[r.any(rm("x",e,t,n),a,s)]}case"ArgMax":{const a=rm("axis",e,t,n);return[r.argMax(rm("x",e,t,n),a)]}case"ArgMin":{const a=rm("axis",e,t,n);return[r.argMin(rm("x",e,t,n),a)]}case"Prod":{const a=rm("axis",e,t,n),s=rm("keepDims",e,t,n);return[r.prod(rm("x",e,t,n),a,s)]}case"Cumprod":{const a=rm("axis",e,t,n),s=rm("exclusive",e,t,n),o=rm("reverse",e,t,n);return[r.cumprod(rm("x",e,t,n),a,s,o)]}case"Cumsum":{const a=rm("axis",e,t,n),s=rm("exclusive",e,t,n),o=rm("reverse",e,t,n);return[r.cumsum(rm("x",e,t,n),a,s,o)]}case"Bincount":const a=rm("x",e,t,n),s=rm("weights",e,t,n),o=rm("size",e,t,n);return[r.bincount(a,s,o)];case"DenseBincount":{const a=rm("x",e,t,n),s=rm("weights",e,t,n),o=rm("size",e,t,n),i=rm("binaryOutput",e,t,n);return[r.denseBincount(a,s,o,i)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"slice_join":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"ConcatV2":case"Concat":{const a=rm("n",e,t,n),s=rm("axis",e,t,n);let o=rm("tensors",e,t,n);return o=o.slice(0,a),[r.concat(o,s)]}case"Gather":{const a=rm("x",e,t,n),s=rm("indices",e,t,n);return[r.gather(a,r.cast(s,"int32"),0)]}case"GatherV2":{const a=rm("axis",e,t,n),s=rm("batchDims",e,t,n),o=rm("x",e,t,n),i=rm("indices",e,t,n);return[r.gather(o,r.cast(i,"int32"),a,s)]}case"Reverse":{const a=rm("dims",e,t,n),s=[];for(let e=0;e<a.length;e++)a[e]&&s.push(e);const o=rm("x",e,t,n);return[r.reverse(o,s)]}case"ReverseV2":{const a=rm("axis",e,t,n),s=rm("x",e,t,n);return[r.reverse(s,a)]}case"Slice":{const a=rm("begin",e,t,n),s=rm("size",e,t,n);return[r.slice(rm("x",e,t,n),a,s)]}case"StridedSlice":{const a=rm("begin",e,t,n),s=rm("end",e,t,n),o=rm("strides",e,t,n),i=rm("beginMask",e,t,n),u=rm("endMask",e,t,n),l=rm("ellipsisMask",e,t,n),c=rm("newAxisMask",e,t,n),p=rm("shrinkAxisMask",e,t,n),d=rm("x",e,t,n);return[r.stridedSlice(d,a,s,o,i,u,l,c,p)]}case"Pack":return Us(()=>{const a=rm("axis",e,t,n),s=rm("tensors",e,t,n),o=s[0].shape,i=r.squeeze(s[0]).shape,u=s.map(e=>{const t=y(e.shape,o);if(!t&&!y(r.squeeze(e).shape,i))throw new Error("the input tensors shape does not match");return t?e:r.reshape(e,o)});return[r.stack(u,a)]});case"Unpack":{const a=rm("axis",e,t,n),s=rm("tensor",e,t,n);return r.unstack(s,a)}case"Tile":{const a=rm("reps",e,t,n);return[r.tile(rm("x",e,t,n),a)]}case"Split":case"SplitV":{const a=rm("axis",e,t,n),s=rm("numOrSizeSplits",e,t,n),o=rm("x",e,t,n);return r.split(o,s,a)}case"ScatterNd":{const a=rm("indices",e,t,n),s=rm("values",e,t,n),o=rm("shape",e,t,n);return[r.scatterND(a,s,o)]}case"GatherNd":{const a=rm("x",e,t,n),s=rm("indices",e,t,n);return[r.gatherND(a,s)]}case"SparseToDense":{const a=rm("sparseIndices",e,t,n),s=rm("outputShape",e,t,n),o=rm("sparseValues",e,t,n),i=rm("defaultValue",e,t,n);return[r.sparseToDense(a,o,s,o.dtype===i.dtype?i:r.cast(i,o.dtype))]}case"TensorScatterUpdate":{const a=rm("indices",e,t,n),s=rm("values",e,t,n),o=rm("tensor",e,t,n);return[r.tensorScatterUpdate(o,a,s)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"sparse":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"SparseFillEmptyRows":{const{outputIndices:a,outputValues:s,emptyRowIndicator:o,reverseIndexMap:i}=r.sparse.sparseFillEmptyRows(rm("indices",e,t,n),rm("values",e,t,n),rm("denseShape",e,t,n),rm("defaultValue",e,t,n));return[a,s,o,i]}case"SparseReshape":{const{outputIndices:a,outputShape:s}=r.sparse.sparseReshape(rm("inputIndices",e,t,n),rm("inputShape",e,t,n),rm("newShape",e,t,n));return[a,s]}case"SparseSegmentMean":return[r.sparse.sparseSegmentMean(rm("data",e,t,n),rm("indices",e,t,n),rm("segmentIds",e,t,n))];case"SparseSegmentSum":return[r.sparse.sparseSegmentSum(rm("data",e,t,n),rm("indices",e,t,n),rm("segmentIds",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"spectral":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"FFT":return[r.fft(rm("x",e,t,n))];case"IFFT":return[r.ifft(rm("x",e,t,n))];case"RFFT":return[r.rfft(rm("x",e,t,n))];case"IRFFT":return[r.irfft(rm("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"string":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"StaticRegexReplace":return[r.string.staticRegexReplace(rm("input",e,t,n),rm("pattern",e,t,n),rm("rewrite",e,t,n),rm("replaceGlobal",e,t,n))];case"StringNGrams":{const{nGrams:a,nGramsSplits:s}=r.string.stringNGrams(rm("data",e,t,n),rm("dataSplits",e,t,n),rm("separator",e,t,n),rm("nGramWidths",e,t,n),rm("leftPad",e,t,n),rm("rightPad",e,t,n),rm("padWidth",e,t,n),rm("preserveShortSequences",e,t,n));return[a,s]}case"StringSplit":{const{indices:a,values:s,shape:o}=r.string.stringSplit(rm("input",e,t,n),rm("delimiter",e,t,n),rm("skipEmpty",e,t,n));return[a,s,o]}case"StringToHashBucketFast":return[r.string.stringToHashBucketFast(rm("input",e,t,n),rm("numBuckets",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"transformation":return a(()=>((e,t,n,r=Gm)=>{switch(e.op){case"Cast":return[r.cast(rm("x",e,t,n),rm("dtype",e,t,n))];case"ExpandDims":{const a=rm("axis",e,t,n);return[r.expandDims(rm("x",e,t,n),a)]}case"Squeeze":{const a=rm("axis",e,t,n);return[r.squeeze(rm("x",e,t,n),a)]}case"Reshape":return[r.reshape(rm("x",e,t,n),rm("shape",e,t,n))];case"EnsureShape":return[r.ensureShape(rm("x",e,t,n),rm("shape",e,t,n))];case"MirrorPad":return[r.mirrorPad(rm("x",e,t,n),rm("padding",e,t,n),rm("mode",e,t,n))];case"PadV2":case"Pad":return[r.pad(rm("x",e,t,n),rm("padding",e,t,n),rm("constantValue",e,t,n))];case"SpaceToBatchND":{const a=rm("blockShape",e,t,n),s=rm("paddings",e,t,n);return[r.spaceToBatchND(rm("x",e,t,n),a,s)]}case"BatchToSpaceND":{const a=rm("blockShape",e,t,n),s=rm("crops",e,t,n);return[r.batchToSpaceND(rm("x",e,t,n),a,s)]}case"DepthToSpace":{const a=rm("blockSize",e,t,n),s=rm("dataFormat",e,t,n).toUpperCase();return[r.depthToSpace(rm("x",e,t,n),a,s)]}case"BroadcastTo":return[r.broadcastTo(rm("x",e,t,n),rm("shape",e,t,n))];case"BroadcastArgs":return[r.broadcastArgs(rm("s0",e,t,n),rm("s1",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n));case"hash_table":return(async(e,t,n,r)=>{switch(e.op){case"HashTable":case"HashTableV2":{const a=r.getHashTableHandleByName(e.name);if(null!=a)return[a];{const a=rm("keyDType",e,t,n),s=rm("valueDType",e,t,n),o=new nf(a,s);return r.addHashTable(e.name,o),[o.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const a=rm("tableHandle",e,t,n,r),s=rm("keys",e,t,n),o=rm("values",e,t,n),i=r.getHashTableById(a.id);return[await i.import(s,o)]}case"LookupTableFind":case"LookupTableFindV2":{const a=rm("tableHandle",e,t,n,r),s=rm("keys",e,t,n),o=rm("defaultValue",e,t,n),i=r.getHashTableById(a.id);return[await i.find(s,o)]}case"LookupTableSize":case"LookupTableSizeV2":{const a=rm("tableHandle",e,t,n,r);return[r.getHashTableById(a.id).tensorSize()]}default:throw TypeError(`Node type ${e.op} is not implemented`)}})(e,t,n,r);case"custom":const s=tm(e.op);if(s&&s.customExecutor)return s.customExecutor(new qm(e,t,n));throw TypeError(`Custom op ${e.op} is not registered.`);default:throw TypeError(`Unknown op '${e.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(e,t,n);return H(s)?s.then(e=>[].concat(e)):[].concat(s)}class af{constructor(e={},t={},n={},r={},a){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=n,this.functionMap=r,this.parseNodeNameCache=a,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let t=0;t<this.contexts.length-1;t++){const n=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(n))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(e=>0===e.id&&0===e.iterationId?"":`${e.frameName}-${e.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(!(this.contexts&&this.contexts.length>1))throw new Error("Cannot exit frame, the context is empty");this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift()}nextIteration(){if(!(this.contexts&&this.contexts.length>0))throw new Error("Cannot increase frame iteration, the context is empty");{this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(const t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sf(e,t,n,r){const a=new Set,s=[];let o=null,i=null;const u=new Set,l=new Set(Object.keys(e).map(e=>um(e)[0]));r=r||[];const c=new Set(r.map(e=>um(e.name)[0])),p=[...t];for(;p.length>0;){const e=p.pop();(df(e)||hf(e)||mf(e))&&null==o&&(o=e,i=o.children.map(e=>e.name).filter(e=>a.has(e))),a.add(e.name),null==n[e.name]&&(l.has(e.name)||c.has(e.name)||(0!==e.inputs.length?e.inputs.forEach(e=>{u.has(e.name)||(u.add(e.name),p.push(e))}):s.push(e.name)))}return{inputs:e,outputs:t,usedNodes:a,missingInputs:s,dynamicNode:o,syncInputs:i}}function of(e,t){const{usedNodes:n,inputs:r}=t,a=Object.keys(r).map(e=>um(e)[0]).map(t=>e.nodes[t]),s=e.initNodes||[],o=e=>n.has("string"==typeof e?e:e.name);function i(e){return[...new Map(e.map(e=>[e.name,e])).values()]}const u=i([...a,...e.weights,...s]).filter(o),l=i([...u,...Object.values(e.nodes)]).filter(o),c=new Map(l.map(e=>[e.name,e])),p={};for(const f of l){p[f.name]=p[f.name]||0;for(const e of f.children)o(e)||(p[e.name]=Number.POSITIVE_INFINITY),p[e.name]=(p[e.name]||0)+1}const d=Object.entries(p).filter(([,e])=>0===e).map(([e])=>e),h=[...d];for(;d.length>0;){const e=d.pop(),t=c.get(e);for(const n of t.children.filter(o))0===--p[n.name]&&(h.push(n.name),d.push(n.name))}const m=function(e,t){const n=new Map(e.map(e=>[e.name,e])),r=t.map(e=>e.name),a=new Set(r);for(;r.length>0;){const e=r.pop(),t=n.get(e);for(const s of t.children)n.has(s.name)&&!a.has(s.name)&&(a.add(s.name),r.push(s.name))}const s=e.filter(e=>a.has(e.name));return s}(h.map(e=>c.get(e)),u);return function(e,t){const n=new Map(e.map((e,t)=>[e.name,t])),r=new Set(t.map(e=>e.name)),a=e=>r.has("string"==typeof e?e:e.name),s=new Set(e.map(e=>e.name)),o=e=>s.has("string"==typeof e?e:e.name);for(const i of e){for(const e of i.children.filter(o)){if(!n.has(e.name))throw new uf(`Child ${e.name} of node ${i.name} is unreachable.`);if(n.get(i.name)>n.get(e.name))throw new uf(`Node ${i.name} is scheduled to run after its child ${e.name}.`)}if(!a(i))for(const e of i.inputs){if(!n.has(e.name))throw new uf(`Input ${e.name} of node ${i.name} is unreachable.`);if(n.get(e.name)>n.get(i.name))throw new uf(`Node ${i.name} is scheduled to run before its input ${e.name}.`)}}}(m,u),m}class uf extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}const lf=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),cf=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),pf=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function df(e){return lf.has(e.op)}function hf(e){return cf.has(e.op)}function mf(e){return pf.has(e.op)}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ff{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const t=Object.keys(e).map(t=>e[t].map(e=>e.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}constructor(e,t){this.graph=e,this.parent=t,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,null!=e.functions&&Object.keys(e.functions).forEach(t=>{this._functionExecutorMap[t]=new ff(e.functions[t],this)})}getCompilationKey(e,t){const n=e.map(e=>e.name).sort(),r=t.map(e=>e.name).sort();return n.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(e,t){const n=sf(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:a,syncInputs:s}=n;if(null!=a)throw new Error(`This execution contains the node '${a.name}', which has the dynamic op '${a.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${s}]`);if(r.length>0){const n=t.map(e=>e.name),a=Object.keys(e);throw new Error(`Cannot compute the outputs [${n}] from the provided inputs [${a}]. Missing the following inputs: [${r}]`)}const o=of(this.graph,n),i=function(e){const t=new Map(e.map((e,t)=>[e.name,t])),n=Number.MAX_SAFE_INTEGER,r=e.map((e,t)=>df(e)?n:t),a=e=>{const n=r[t.get(e.name)];return null==n?-1:n},s=e.map((e,t)=>e.children.map(a).reduce((e,t)=>Math.max(e,t),r[t])),o=new Map;for(let i=0;i<e.length;++i){const t=s[i];if(t===n)continue;const r=e[i],a=e[t];o.has(a.name)||o.set(a.name,[]),o.get(a.name).push(r)}return o}(o);return{orderedNodes:o,nodeLiveUntilMap:i}}cloneAndKeepTensor(e){if(null==e)return null;const t=e.clone();return Ws(t),t}cloneTensorList(e){if(!e)return null;return e.map(e=>this.cloneAndKeepTensor(e))}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([e,t])=>[e,this.cloneTensorList(t)]))}execute(e,t){this.disposeIntermediateTensors(),e=this.mapInputs(e);const n=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);const r=n.map(e=>this.graph.nodes[um(e)[0]]),a=t.map(e=>um(e)[0]),s=new Set(a);let o=a.map(e=>this.graph.nodes[e]);0===o.length&&(o=this._outputs);const i=this.getCompilationKey(r,o);let u=this.compiledMap.get(i);null==u&&(u=this.compile(e,o),this.compiledMap.set(i,u));try{this.keepIntermediateTensors=Y().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(vf){this.keepIntermediateTensors=!1}const l={},c={};return Us(()=>{const n=new af(this.weightMap,l,c,this.functionExecutorMap,this.parseNodeNameCache),r=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(t=>{const[a,s]=um(t,n),o=[];o[s]=e[t],r[a]=o,this.keepIntermediateTensors&&(this.clonedTensorsMap[a]=this.cloneTensorList(o))});const a=this.getFrozenTensorIds(r),{orderedNodes:o,nodeLiveUntilMap:i}=u;for(const e of o){if(r[e.name])continue;const t=rf(e,r,n,this._resourceManager);if(H(t))throw new Error(`The execution of the op '${e.op}' returned a promise. Please use model.executeAsync() instead.`);r[e.name]=t,this.keepIntermediateTensors&&(this.clonedTensorsMap[e.name]=this.cloneTensorList(t)),this.checkTensorForDisposalWithNodeLiveUntilInfo(e,r,n,a,s,i.get(e.name))}return null==this.parent&&n.dispose(a),t.map(e=>am(e,r,n))})}getFrozenTensorIds(e){const t=[].concat.apply([],Object.keys(e).map(t=>e[t]).map(e=>e.map(e=>e.id)));return new Set(t)}checkTensorForDisposal(e,t,n,r,a,s,o){if(!df(t)&&!s.has(e)){for(const r of n[e])null!=r&&(o[r.id]=(o[r.id]||0)+t.children.length);for(const e of t.inputs){if(df(e))continue;const t=sm(e.name,n,r);if(null!=t)for(const e of t){if(!e||e.kept||a.has(e.id))continue;const t=o[e.id];1===t?(e.dispose(),delete o[e.id]):null!=t&&o[e.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,t,n,r,a,s){function o(e){return df(e)||a.has(e.name)}if(!df(e)&&null!=s)for(const i of s){if(o(i))continue;const e=sm(i.name,t,n);for(const t of e)!t||t.kept||r.has(t.id)||t.dispose()}}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const t of e)t&&!t.isDisposed&&t.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,t,n=!1,r={},a={}){this.disposeIntermediateTensors(),n||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepIntermediateTensors=Y().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(vf){this.keepIntermediateTensors=!1}const s=new af(this.weightMap,r,a,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const o=await this.executeWithControlFlow(e,s,t,n),i=t.map(e=>am(e,o,s)),u=i.map(e=>e.id),l=Object.keys(e).map(t=>e[t].id),c=new Set([...u,...l,...this.weightIds]);return Object.values(o).forEach(e=>{e.forEach(e=>{!e||e.isDisposed||c.has(e.id)||e.dispose()})}),null==this.parent&&s.dispose(c),i}async executeFunctionAsync(e,t,n){const r=e.reduce((e,t,n)=>(e[this.inputs[n].name]=t,e),{});return this._executeAsync(r,this.outputNodes,!0,t,n)}async executeWithControlFlow(e,t,n,r){const a=Object.keys(e),s=a.map(e=>this.graph.nodes[um(e)[0]]),o=n.map(e=>um(e)[0]),i=new Set(o);let u=o.map(e=>this.graph.nodes[e]);0===u.length&&(u=this._outputs);const{usedNodes:l,missingInputs:c,dynamicNode:p,syncInputs:d}=sf(e,u,this.weightMap,this._initNodes),h=[...s,...this.graph.weights,...this._initNodes||[]].map(e=>({node:e,contexts:t.currentContext})),m=Object.assign({},this.weightMap);Object.keys(e).forEach(t=>{const[n,r]=um(t),a=[];a[r]=e[t],m[n]=a});const f={},g=this.getFrozenTensorIds(m),y={};for(;h.length>0;){const e=this.processStack(s,h,t,m,y,g,i,f,l);await Promise.all(e)}const b=u.filter(e=>!df(e)&&!am(e.name,m,t)).map(e=>e.name);if(b.length>0){let e="";throw null!=p&&(e=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${d}]`),new Error(`Cannot compute the outputs [${b}] from the provided inputs [${a}]. Consider providing the following inputs: [${c}]. ${e}`)}return m}processStack(e,t,n,r,a,s,o,i,u){const l=[];for(;t.length>0;){const e=t.pop();n.currentContext=e.contexts;let c="";if("Enter"===e.node.op&&rm("isConstant",e.node,r,n)&&([c]=om(e.node.name,n)),null==r[e.node.name]){const p=rf(e.node,r,n,this._resourceManager);c||([c]=om(e.node.name,n));const d=n.currentContext;H(p)?l.push(p.then(l=>(r[c]=l,this.keepIntermediateTensors&&(this.clonedTensorsMap[c]=this.cloneTensorList(l)),n.currentContext=d,this.checkTensorForDisposal(c,e.node,r,n,s,o,i),this.processChildNodes(e.node,t,n,r,a,u),l))):(r[c]=p,this.keepIntermediateTensors&&(this.clonedTensorsMap[c]=this.cloneTensorList(p)),this.checkTensorForDisposal(c,e.node,r,n,s,o,i),this.processChildNodes(e.node,t,n,r,a,u))}else this.processChildNodes(e.node,t,n,r,a,u)}return l}processChildNodes(e,t,n,r,a,s){e.children.forEach(e=>{const[o]=om(e.name,n);!a[o]&&s.has(e.name)&&("Merge"===e.op?e.inputNames.some(e=>!!am(e,r,n))&&(a[o]=!0,t.push({contexts:n.currentContext,node:e})):e.inputNames.every(e=>!!am(e,r,n))&&(a[o]=!0,t.push({contexts:n.currentContext,node:e})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(e=>e.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{const n=e[t],[r]=um(t),a=this.graph.nodes[r];if(a.attrParams.shape&&a.attrParams.shape.value){const e=a.attrParams.shape.value;d(e.length===n.shape.length&&n.shape.every((t,n)=>-1===e[n]||e[n]===t),()=>`The shape of dict['${a.name}'] provided in model.execute(dict) must be [${e}], but was [${n.shape}]`)}a.attrParams.dtype&&a.attrParams.dtype.value&&d(n.dtype===a.attrParams.dtype.value,()=>`The dtype of dict['${a.name}'] provided in model.execute(dict) must be ${a.attrParams.dtype.value}, but was ${n.dtype}`)})}mapInputs(e){var t,n;const r={};for(const a in e){const s=null===(n=null===(t=this._signature)||void 0===t?void 0:t.inputs)||void 0===n?void 0:n[a];null!=s?r[s.name]=e[a]:r[a]=e[a]}return r}checkInputs(e){const t=Object.keys(e).filter(e=>{const[t]=um(e);return null==this.graph.nodes[t]});if(t.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(e=>{var t,n;const r=null===(n=null===(t=this._signature)||void 0===t?void 0:t.outputs)||void 0===n?void 0:n[e];return null!=r?r.name:e},{})}checkOutputs(e){e.forEach(e=>{const[t]=um(e);if(!this.graph.nodes[t])throw new Error(`The output '${e}' is not found in the graph`)})}}class gf{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}
/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yf="?tfjs-format=file",bf="model.json";class wf{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,t={},n=Ph){this.modelUrl=e,this.loadOptions=t,this.version="n/a",this.io=n,null==t&&(this.loadOptions={}),this.resourceManager=new gf}findIOHandler(){const e=this.modelUrl;if(null!=e.load)this.handler=e;else if(null!=this.loadOptions.requestInit)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const t=this.io.getLoadHandlers(e,this.loadOptions);if(0===t.length)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw new Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),null==this.handler.load)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return H(e)?e.then(e=>null==e.getWeightStream?this.loadSync(e):this.loadStreaming(e)):this.loadSync(e)}loadSync(e){const t=this.io.decodeWeights(e.weightData,e.weightSpecs);return this.loadWithWeightMap(e,t)}async loadStreaming(e){if(null==e.getWeightStream)throw new Error("Model artifacts missing streamWeights function");const t=await io(e.getWeightStream(),e.weightSpecs);return this.loadWithWeightMap(e,t)}loadWithWeightMap(e,t){this.artifacts=e;const n=this.artifacts.modelTopology;let r=this.artifacts.signature;if(null!=this.artifacts.userDefinedMetadata){const e=this.artifacts.userDefinedMetadata;null!=e.signature&&(r=e.signature),null!=e.structuredOutputKeys&&(this.structuredOutputKeys=e.structuredOutputKeys)}if(this.signature=r,this.version=`${n.versions.producer}.${n.versions.minConsumer}`,this.executor=new ff(Mm.Instance.transformGraph(n,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(t),this.executor.resourceManager=this.resourceManager,null!=e.modelInitializer&&null!=e.modelInitializer.node){const t=Mm.Instance.transformGraph(e.modelInitializer);this.initializer=new ff(t),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,t){if("string"==typeof e){const t=this.io.getSaveHandlers(e);if(0===t.length)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(t.length>1)throw new Error(`Found more than one (${t.length}) save handlers for URL '${e}'`);e=t[0]}if(null==e.save)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const t={};return(e instanceof Va?[e]:e).forEach((e,n)=>t[this.structuredOutputKeys[n]]=e),t}return e}predict(e,t){const n=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(n)}async predictAsync(e,t){const n=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(n)}normalizeInputs(e){var t;if(!(e instanceof Va||Array.isArray(e))){const n=null===(t=this.signature)||void 0===t?void 0:t.inputs;if(null!=n)for(const t in n){const r=n[t];null!=r.resourceId&&(e[t]=this.resourceIdToCapturedInput[r.resourceId])}return e}e=Array.isArray(e)?e:[e];const n=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+n!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-n} non-resource placeholders, while there are ${e.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((t,n)=>{var a,s,o;const i=null===(o=null===(s=null===(a=this.signature)||void 0===a?void 0:a.inputs)||void 0===s?void 0:s[n])||void 0===o?void 0:o.resourceId;return t[n]=null!=i?this.resourceIdToCapturedInput[i]:e[r++],t},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return null==this.initializer?[]:null==this.initializerSignature?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return null==this.initializer?[]:null==this.initializerSignature?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const t=this.initializerSignature.outputs,n=Object.keys(t);for(let r=0;r<n.length;r++){const a=t[n[r]];this.resourceIdToCapturedInput[a.resourceId]=e[r]}}}execute(e,t){null==this.resourceIdToCapturedInput&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const n=this.executor.execute(e,t);return n.length>1?n:n[0]}async executeAsync(e,t){null==this.resourceIdToCapturedInput&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const n=await this.executor.executeAsync(e,t);return n.length>1?n:n[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,n)=>(t[n]=[e[n]],t),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&js(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function Nf(e,t={},n=Ph){if(null==e)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");null==t&&(t={}),t.fromTFHub&&"string"==typeof e&&(e=function(e){e.endsWith("/")||(e+="/");return`${e}${bf}${yf}`}(e));const r=new wf(e,t,n);return await r.load(),r}function xf(e){if(null==e)throw new Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let t;if(e instanceof Array){const[n,r]=e;if(!n)throw new Error("modelJSON must be the first element of the array");if(!(r&&r instanceof ArrayBuffer))throw new Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in n))throw new Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in n))throw new Error("Model JSON is missing 'weightsManifest'");t=Lh(fo(n,bo(n.weightsManifest),r))}else if("load"in e)t=e;else{if(!("modelTopology"in e&&"weightSpecs"in e&&"weightData"in e))throw new Error("Unknown model format");t=Lh(e)}const n=new wf(t);return n.load(),n}export{nd as $,bh as A,Ea as B,ad as C,vu as D,fs as E,Bu as F,Du as G,Dp as H,Oi as I,Cu as J,xi as K,vi as L,xh as M,Ti as N,bi as O,Ru as P,wi as Q,vh as R,Nh as S,Va as T,Ni as U,$i as V,Ai as W,Fu as X,zu as Y,Nu as Z,rd as _,O as a,Fe as a$,Pu as a0,xu as a1,Lu as a2,Tr as a3,sd as a4,Mi as a5,Ii as a6,ts as a7,Op as a8,$p as a9,me as aA,fe as aB,Ne as aC,we as aD,be as aE,ye as aF,xe as aG,ve as aH,oc as aI,ke as aJ,_e as aK,Ie as aL,Ae as aM,ku as aN,Ml as aO,sl as aP,dl as aQ,$e as aR,Oe as aS,T as aT,yp as aU,De as aV,td as aW,ou as aX,Re as aY,au as aZ,Be as a_,vr as aa,kd as ab,Ed as ac,_d as ad,jp as ae,re as af,ai as ag,vp as ah,ae as ai,Gu as aj,qu as ak,_l as al,Sl as am,ri as an,se as ao,oe as ap,Ku as aq,Di as ar,ie as as,ce as at,Eu as au,pe as av,de as aw,he as ax,ti as ay,ge as az,_a as b,pn as b$,lu as b0,Le as b1,ip as b2,Pe as b3,up as b4,Ue as b5,mu as b6,Ge as b7,id as b8,ud as b9,At as bA,Rt as bB,zt as bC,Bt as bD,Su as bE,Pt as bF,Vt as bG,pl as bH,qt as bI,Wt as bJ,jt as bK,Ut as bL,Kt as bM,jl as bN,Ht as bO,Jt as bP,Zt as bQ,Li as bR,Xt as bS,nl as bT,Qt as bU,en as bV,on as bW,sn as bX,un as bY,Pp as bZ,ln as b_,Ze as ba,Ye as bb,Xe as bc,nt as bd,rt as be,at as bf,Xu as bg,ot as bh,it as bi,ut as bj,dt as bk,ht as bl,mt as bm,np as bn,el as bo,ft as bp,xp as bq,Lp as br,bt as bs,wt as bt,vt as bu,Tt as bv,St as bw,kt as bx,al as by,Mt as bz,vs as c,fi as c$,fl as c0,ju as c1,dn as c2,hn as c3,hu as c4,et as c5,wn as c6,En as c7,Nn as c8,xn as c9,Vn as cA,Yn as cB,Xn as cC,gr as cD,ar as cE,Un as cF,sr as cG,or as cH,ir as cI,cr as cJ,dr as cK,hr as cL,Pl as cM,rl as cN,Yu as cO,fr as cP,Mr as cQ,Ua as cR,si as cS,oi as cT,ii as cU,li as cV,ci as cW,pi as cX,di as cY,hi as cZ,mi as c_,Sn as ca,kn as cb,vn as cc,Tn as cd,_n as ce,Zc as cf,In as cg,An as ch,Dn as ci,$l as cj,Cn as ck,Ln as cl,zn as cm,Fn as cn,pu as co,Bn as cp,du as cq,Rn as cr,tc as cs,qn as ct,Pn as cu,zi as cv,jn as cw,Ui as cx,Wn as cy,Fi as cz,y as d,Pi as d$,gi as d0,yi as d1,Ci as d2,ji as d3,Zi as d4,Xi as d5,Qi as d6,su as d7,iu as d8,gu as d9,Vl as dA,Uu as dB,ql as dC,Gl as dD,Kl as dE,Ju as dF,Xl as dG,Ql as dH,ic as dI,uc as dJ,lc as dK,Kc as dL,Hc as dM,Jc as dN,Bd as dO,zd as dP,bp as dQ,tp as dR,rp as dS,ap as dT,op as dU,hp as dV,kl as dW,wp as dX,Np as dY,Tp as dZ,Sp as d_,yu as da,wu as db,_u as dc,Iu as dd,Mu as de,Ou as df,Zu as dg,Qu as dh,mp as di,ni as dj,fp as dk,gp as dl,il as dm,ul as dn,ll as dp,cl as dq,ml as dr,El as ds,Il as dt,Al as du,gl as dv,Ol as dw,Dl as dx,Bl as dy,Vu as dz,ka as e,k as e$,Fp as e0,zp as e1,Ta as e2,Qs as e3,dp as e4,pp as e5,cp as e6,lp as e7,kp as e8,Vc as e9,xo as eA,Rh as eB,Ah as eC,no as eD,cu as eE,od as eF,uu as eG,Hl as eH,Wi as eI,qi as eJ,Gi as eK,zl as eL,Ri as eM,$s as eN,qc as eO,fu as eP,H as eQ,us as eR,is as eS,Fc as eT,Ep as eU,Sa as eV,s as eW,a as eX,Ls as eY,V as eZ,Xo as e_,cd as ea,ru as eb,nu as ec,tu as ed,eu as ee,Xp as ef,jc as eg,Bp as eh,tl as ei,sh as ej,ph as ek,dh as el,hh as em,Up as en,js as eo,Y as ep,C as eq,Ws as er,Yi as es,Qo as et,fh as eu,i as ev,Ps as ew,No as ex,to as ey,po as ez,Gh as f,br as f$,K as f0,G as f1,Me as f2,j as f3,bn as f4,va as f5,A as f6,Se as f7,E as f8,st as f9,Je as fA,P as fB,W as fC,Qe as fD,tt as fE,lt as fF,ct as fG,R as fH,pt as fI,Nr as fJ,xr as fK,gt as fL,Nt as fM,It as fN,$t as fO,Ot as fP,Dt as fQ,Gt as fR,Yt as fS,nn as fT,rn as fU,an as fV,h as fW,mn as fX,fn as fY,gn as fZ,yn as f_,yt as fa,Et as fb,_t as fc,xa as fd,tn as fe,za as ff,Qn as fg,Na as fh,c as fi,Ys as fj,v as fk,wr as fl,ue as fm,le as fn,Te as fo,Ee as fp,xt as fq,Ce as fr,ze as fs,je as ft,Ve as fu,U as fv,We as fw,qe as fx,Ke as fy,He as fz,Ia as g,Zs as g$,Mn as g0,On as g1,Gn as g2,Kn as g3,Hn as g4,Jn as g5,Zn as g6,er as g7,tr as g8,nr as g9,Wa as gA,qd as gB,mr as gC,ja as gD,ui as gE,Vi as gF,Ki as gG,Hi as gH,qp as gI,Ji as gJ,Tu as gK,Kh as gL,As as gM,Dr as gN,Qp as gO,vl as gP,Bs as gQ,nm as gR,Ns as gS,bu as gT,Fs as gU,zs as gV,Au as gW,Rs as gX,Cs as gY,Yp as gZ,$u as g_,rr as ga,$n as gb,ur as gc,lr as gd,pr as ge,Ar as gf,w as gg,N as gh,l as gi,S as gj,bs as gk,x as gl,p as gm,M as gn,ws as go,b as gp,ns as gq,yr as gr,ee as gs,Z as gt,wf as gu,Ct as gv,Ft as gw,Lt as gx,_s as gy,cn as gz,ks as h,$r as h$,Xs as h0,pd as h1,Zp as h2,Hs as h3,_r as h4,Er as h5,Ir as h6,yl as h7,bl as h8,ol as h9,Gc as hA,em as hB,Js as hC,Xc as hD,Yc as hE,Qc as hF,ep as hG,Hp as hH,Cp as hI,Rl as hJ,mh as hK,Gs as hL,eo as hM,sp as hN,rh as hO,ih as hP,Jp as hQ,nh as hR,uh as hS,_p as hT,Ip as hU,Ap as hV,Mp as hW,Rp as hX,cs as hY,qs as hZ,Or as h_,ed as ha,Ph as hb,hl as hc,xf as hd,oh as he,Fl as hf,Ll as hg,Wl as hh,Kp as hi,Jl as hj,Zl as hk,ec as hl,nc as hm,rc as hn,ac as ho,sc as hp,ei as hq,Vs as hr,cc as hs,pc as ht,dc as hu,hc as hv,Pc as hw,Uc as hx,Wc as hy,Ks as hz,ah as i,Vp as i0,Aa as i1,wl as i2,Nl as i3,xl as i4,Wp as i5,d as j,Yl as k,Nf as l,Yo as m,Gp as n,Is as o,Bi as p,f as q,z as r,Wu as s,Us as t,u,gh as v,wh as w,yh as x,B as y,Ul as z};
