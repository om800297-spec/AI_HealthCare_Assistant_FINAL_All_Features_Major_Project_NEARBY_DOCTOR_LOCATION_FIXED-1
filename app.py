from flask import Flask,render_template,request,redirect,url_for,session,jsonify,flash
import sqlite3,os,re,math,requests
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from datetime import datetime
app=Flask(__name__); app.secret_key=os.getenv("SECRET_KEY","demo-secret")
DB="healthcare.db"
UPLOAD_DIR="uploads"
ALLOWED_EXTENSIONS={"pdf"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Explainable ML-style triage engine using synthetic training examples.
DATA=[
("fever cough sore throat fatigue","Routine"),("runny nose cough mild fever","Routine"),
("headache fatigue nausea","Routine"),("mild diarrhea stomach pain","Routine"),
("persistent fever severe weakness","Priority"),("high fever dehydration dizziness","Priority"),
("repeated vomiting severe abdominal pain","Priority"),("confusion severe weakness fever","Priority"),
("chest pain sweating breathing difficulty","Emergency"),("very low oxygen breathing difficulty","Emergency"),
("unconscious seizure","Emergency"),("stroke facial weakness speech difficulty","Emergency"),
("severe bleeding unconscious","Emergency")]
def tok(s): return re.findall(r"[a-z]+",s.lower())
class Triage:
 def __init__(self):
  self.cls=sorted(set(y for _,y in DATA)); self.v=set(); self.w={c:{} for c in self.cls}; self.n={c:0 for c in self.cls}; self.docs={c:0 for c in self.cls}
  for x,c in DATA:
   ws=tok(x); self.docs[c]+=1; self.n[c]+=len(ws)
   for q in ws:self.v.add(q); self.w[c][q]=self.w[c].get(q,0)+1
 def predict(self,text,age):
  ws=tok(text); em={"chest pain","unconscious","seizure","stroke","severe bleeding","breathing difficulty"}
  if any(x in text.lower() for x in em):
   return "Emergency",95,"Emergency-pattern keywords detected","Seek immediate emergency medical care."
  scores={}
  for c in self.cls:
   s=math.log((self.docs[c]+1)/(len(DATA)+len(self.cls)))
   for q in ws:s+=math.log((self.w[c].get(q,0)+1)/(self.n[c]+len(self.v)))
   scores[c]=s
  best=max(scores,key=scores.get); ex={k:math.exp(v-max(scores.values())) for k,v in scores.items()}
  conf=round(100*ex[best]/sum(ex.values()),1)
  return best,conf,f"Explainable model category: {best}. Keywords considered: {', '.join(ws[:12]) or 'none'}",{
   "Routine":"Monitor symptoms and consult a clinician if they persist or worsen.",
   "Priority":"Arrange a clinician consultation soon, especially if symptoms worsen.",
   "Emergency":"Seek immediate emergency medical care."
  }[best]
ai=Triage()


SPECIALTY_RULES=[
 (['chest pain','palpitation','heart','chest pressure'],'Cardiology'),
 (['breathing difficulty','shortness of breath','wheezing','persistent cough','asthma'],'Pulmonology'),
 (['sore throat','ear pain','sinus','tonsil','hearing'],'ENT'),
 (['rash','itching','acne','skin','eczema'],'Dermatology'),
 (['abdominal pain','vomiting','nausea','diarrhea','gastric','stomach'],'Gastroenterology'),
 (['joint pain','back pain','bone','arthritis'],'Orthopedics'),
 (['headache','migraine','seizure','numbness'],'Neurology'),
 (['anxiety','depression','panic','sleep'],'Psychiatry')
]
def infer_department(symptoms):
    t=(symptoms or '').lower()
    for terms,d in SPECIALTY_RULES:
        if any(x in t for x in terms): return d
    return 'General Medicine'

CITY_COORDS = {
    "jabalpur": (23.1815, 79.9864),
    "jabalpur madhya pradesh": (23.1815, 79.9864),
    "indore": (22.7196, 75.8577),
    "bhopal": (23.2599, 77.4126),
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "pune": (18.5204, 73.8567),
    "nagpur": (21.1458, 79.0882),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319),
    "varanasi": (25.3176, 82.9739),
    "agra": (27.1767, 78.0081),
    "gwalior": (26.2183, 78.1828),
    "rewa": (24.5362, 81.3037),
}

def geocode_place(place):
    """Resolve a city/area without making the feature depend entirely on geocoding APIs."""
    raw=(place or '').strip()
    if not raw:
        return None
    key=re.sub(r'\s+', ' ', raw.lower()).strip()
    # First use reliable built-in coordinates for common demo/test cities.
    for name,coords in CITY_COORDS.items():
        if key == name or key.startswith(name + ',') or name in key:
            return (*coords, raw)
    # Then try Nominatim as a fallback for other cities/areas.
    try:
        rr=requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q':raw,'format':'json','limit':1},
            headers={'User-Agent':'HealthAI-MajorProject/6.0 (educational project)'},
            timeout=12
        )
        rr.raise_for_status()
        data=rr.json()
        if data:
            return (float(data[0]['lat']), float(data[0]['lon']), data[0].get('display_name',raw))
    except Exception:
        pass
    return None

def nearby_providers(lat,lon,department,radius=25000):
    """Find nearby medical providers without falsely labeling hospitals as specialists."""
    endpoints=[
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
        'https://overpass.private.coffee/api/interpreter'
    ]
    keys={'Cardiology':['cardio','cardiology','heart','cardiac'],
          'Dermatology':['derma','dermatology','skin'],
          'ENT':['ent','otorhinolaryng','ear','nose','throat'],
          'Pulmonology':['pulmon','pulmonology','respir','lung'],
          'Gastroenterology':['gastro','gastroenterology','digest'],
          'Orthopedics':['ortho','orthopedic','orthopaedic','bone','musculoskeletal'],
          'Neurology':['neuro','neurology'],
          'Psychiatry':['psych','psychiatry','mental'],
          'General Medicine':['general','physician','medicine']}
    wanted=keys.get(department,[])

    def run_query(r):
        q="""[out:json][timeout:45];(
          node[\"healthcare\"=\"specialist\"](around:%s,%s,%s);
          way[\"healthcare\"=\"specialist\"](around:%s,%s,%s);
          node[\"amenity\"=\"doctors\"](around:%s,%s,%s);
          way[\"amenity\"=\"doctors\"](around:%s,%s,%s);
          node[\"healthcare\"=\"doctor\"](around:%s,%s,%s);
          way[\"healthcare\"=\"doctor\"](around:%s,%s,%s);
          node[\"amenity\"=\"clinic\"](around:%s,%s,%s);
          way[\"amenity\"=\"clinic\"](around:%s,%s,%s);
          node[\"healthcare\"=\"clinic\"](around:%s,%s,%s);
          way[\"healthcare\"=\"clinic\"](around:%s,%s,%s);
        );out center tags;""" % tuple([r,lat,lon]*10)
        last=None
        for endpoint in endpoints:
            try:
                rr=requests.post(endpoint,data={'data':q},timeout=50,headers={'User-Agent':'HealthAI-MajorProject/5.0'})
                rr.raise_for_status(); return rr.json()
            except Exception as e:
                last=e
                try:
                    rr=requests.get(endpoint,params={'data':q},timeout=50,headers={'User-Agent':'HealthAI-MajorProject/5.0'})
                    rr.raise_for_status(); return rr.json()
                except Exception as e2: last=e2
        raise RuntimeError('All nearby-provider map services failed') from last

    def collect(data, strict=True):
        items=[]
        for e in data.get('elements',[]):
            tags=e.get('tags',{}); name=tags.get('name') or tags.get('official_name')
            if not name: continue
            plat=e.get('lat',e.get('center',{}).get('lat')); plon=e.get('lon',e.get('center',{}).get('lon'))
            if plat is None or plon is None: continue
            if tags.get('amenity')=='hospital' or tags.get('healthcare')=='hospital': continue
            txt=' '.join(str(tags.get(k,'')) for k in ['name','healthcare:speciality','healthcare:specialty','medical:speciality','medical:specialty','speciality','specialty','description','operator']).lower()
            explicit=any(k in txt for k in wanted)
            is_doctor=tags.get('amenity')=='doctors' or tags.get('healthcare') in ('doctor','specialist')
            is_clinic=tags.get('amenity')=='clinic' or tags.get('healthcare')=='clinic'
            if strict and department!='General Medicine' and not explicit: continue
            if not (is_doctor or is_clinic): continue
            dist=math.sqrt(((float(plat)-float(lat))*111)**2+((float(plon)-float(lon))*111*math.cos(math.radians(float(lat))))**2)
            items.append({'name':name,'department':department,'address':', '.join([tags.get('addr:housenumber',''),tags.get('addr:street',''),tags.get('addr:city','')]).strip(', ') or 'Address not listed on OpenStreetMap','distance_km':round(dist,2),'phone':tags.get('phone') or tags.get('contact:phone') or 'Not listed','map_url':'https://www.google.com/maps/search/?api=1&query=%s,%s'%(plat,plon),'specialty_verified':bool(explicit),'score':10 if explicit else 1})
        items.sort(key=lambda x:(-x['score'],x['distance_km']))
        out=[]; seen=set()
        for x in items:
            k=x['name'].strip().lower()
            if k in seen: continue
            seen.add(k); out.append(x)
            if len(out)>=10: break
        return out

    for r in (25000,50000):
        data=run_query(r)
        verified=collect(data, strict=True)
        if verified: return verified
    for r in (25000,50000):
        data=run_query(r)
        fallback=collect(data, strict=False)
        if fallback: return fallback
    return []

DOCTORS = [
 {"name":"Dr. Ananya Sharma","department":"General Medicine","specialties":"Fever, cough, fatigue, general symptoms","experience":"8 years","mode":"Clinic / Video"},
 {"name":"Dr. Rohan Mehta","department":"Cardiology","specialties":"Chest discomfort and heart-related concerns","experience":"12 years","mode":"Clinic / Video"},
 {"name":"Dr. Neha Verma","department":"Dermatology","specialties":"Rash, itching, acne and skin concerns","experience":"7 years","mode":"Clinic / Video"},
 {"name":"Dr. Arjun Patel","department":"ENT","specialties":"Sore throat, ear, nose and throat concerns","experience":"9 years","mode":"Clinic / Video"},
 {"name":"Dr. Priya Nair","department":"Gastroenterology","specialties":"Abdominal pain, nausea, vomiting and digestion","experience":"10 years","mode":"Clinic / Video"},
 {"name":"Dr. Kavya Singh","department":"Pulmonology","specialties":"Cough and respiratory concerns","experience":"11 years","mode":"Clinic / Video"}
]
def recommend_doctors(symptoms):
    dept=infer_department(symptoms)
    return dept,[x for x in DOCTORS if x['department']==dept]

def db():
 c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
def init():
 c=db(); c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,name TEXT,email TEXT UNIQUE,password TEXT,role TEXT,created TEXT);
 CREATE TABLE IF NOT EXISTS assessments(id INTEGER PRIMARY KEY,user_id INTEGER,symptoms TEXT,risk TEXT,score REAL,explanation TEXT,action TEXT,created TEXT);
 CREATE TABLE IF NOT EXISTS vitals(id INTEGER PRIMARY KEY,user_id INTEGER,hr REAL,sys REAL,dia REAL,temp REAL,spo2 REAL,created TEXT);
 CREATE TABLE IF NOT EXISTS appointments(id INTEGER PRIMARY KEY,user_id INTEGER,doctor TEXT,department TEXT,date TEXT,reason TEXT,status TEXT,created TEXT);
 CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY,user_id INTEGER,text TEXT,flags TEXT,created TEXT);
 CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY,user_id INTEGER,level TEXT,message TEXT,created TEXT);"""); c.commit(); c.close()
def user():
 if "uid" not in session:return None
 c=db();u=c.execute("SELECT * FROM users WHERE id=?",(session["uid"],)).fetchone();c.close();return u
@app.context_processor
def ctx():return {"me":user()}
@app.route("/")
def home():return redirect("/dashboard") if user() else render_template("landing.html")
@app.route("/register",methods=["GET","POST"])
def register():
 if request.method=="POST":
  c=db()
  try:
   c.execute("INSERT INTO users(name,email,password,role,created) VALUES(?,?,?,?,?)",(request.form["name"],request.form["email"].lower(),request.form["password"],"patient",datetime.now().isoformat(timespec="seconds")));c.commit()
   uid=c.execute("SELECT id FROM users WHERE email=?",(request.form["email"].lower(),)).fetchone()["id"];c.close();session["uid"]=uid;return redirect("/dashboard")
  except sqlite3.IntegrityError:c.close();flash("Email already registered")
 return render_template("register.html")
@app.route("/login",methods=["GET","POST"])
def login():
 if request.method=="POST":
  c=db();u=c.execute("SELECT * FROM users WHERE email=? AND password=?",(request.form["email"].lower(),request.form["password"])).fetchone();c.close()
  if u:session["uid"]=u["id"];return redirect("/dashboard")
  flash("Invalid login")
 return render_template("login.html")
@app.route("/logout")
def logout():session.clear();return redirect("/")


@app.route("/doctors")
def doctors():
    if not user(): return redirect("/login")
    symptoms=request.args.get("symptoms","")
    dept,docs=recommend_doctors(symptoms)
    return render_template("doctors.html",symptoms=symptoms,department=dept,doctors=docs)

@app.route('/nearby-doctors',methods=['GET','POST'])
def nearby_doctors():
    if not user(): return redirect('/login')
    symptoms=request.values.get('symptoms','')
    department=infer_department(symptoms)
    lat=request.values.get('lat',''); lon=request.values.get('lon',''); place=request.values.get('place','').strip()
    providers=[]; error=None; location_name=''
    if (not lat or not lon) and place:
        try:
            geo=geocode_place(place)
            if geo:
                lat,lon,location_name=geo
            else:
                error='Location not found. Try a city, area, or PIN code.'
        except Exception:
            error='Location lookup failed. Check internet access and try again.'
    if lat and lon:
        try:
            providers=nearby_providers(float(lat),float(lon),department)
            if not providers: error='No mapped specialist doctors/clinics were found within 50 km. Try another nearby area.'
        except Exception:
            error='Nearby search is temporarily unavailable. Check internet access and try again.'
    return render_template('nearby_doctors.html',symptoms=symptoms,department=department,providers=providers,error=error,lat=lat,lon=lon,place=place,location_name=location_name)

@app.route("/dashboard")
def dashboard():
 u=user()
 if not u:return redirect("/login")
 c=db();A=c.execute("SELECT * FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT 6",(u["id"],)).fetchall();V=c.execute("SELECT * FROM vitals WHERE user_id=? ORDER BY id DESC LIMIT 6",(u["id"],)).fetchall();P=c.execute("SELECT * FROM appointments WHERE user_id=? ORDER BY id DESC LIMIT 6",(u["id"],)).fetchall();L=c.execute("SELECT * FROM alerts WHERE user_id=? ORDER BY id DESC LIMIT 5",(u["id"],)).fetchall();c.close()
 return render_template("dashboard.html",A=A,V=V,P=P,L=L)
@app.route("/triage",methods=["GET","POST"])
def triage():
 u=user()
 if not u:return redirect("/login")
 r=None
 if request.method=="POST":
  risk,score,ex,action=ai.predict(request.form["symptoms"],int(request.form.get("age") or 0));r=(risk,score,ex,action)
  c=db();c.execute("INSERT INTO assessments(user_id,symptoms,risk,score,explanation,action,created) VALUES(?,?,?,?,?,?,?)",(u["id"],request.form["symptoms"],risk,score,ex,action,datetime.now().isoformat(timespec="seconds")))
  if risk=="Emergency":c.execute("INSERT INTO alerts(user_id,level,message,created) VALUES(?,?,?,?)",(u["id"],"CRITICAL","Emergency-pattern detected; seek professional emergency care.",datetime.now().isoformat(timespec="seconds")))
  c.commit();c.close()
 dept,docs=recommend_doctors(request.form.get("symptoms","")) if request.method=="POST" else ("General Medicine",[])
 return render_template("triage.html",r=r,doctor_department=dept,doctors=docs)
@app.route("/vitals",methods=["GET","POST"])
def vitals():
 u=user()
 if not u:return redirect("/login")
 if request.method=="POST":
  c=db();vals=[request.form.get(x) or None for x in ["hr","sys","dia","temp","spo2"]];c.execute("INSERT INTO vitals(user_id,hr,sys,dia,temp,spo2,created) VALUES(?,?,?,?,?,?,?)",(u["id"],*vals,datetime.now().isoformat(timespec="seconds")));c.commit();c.close();flash("Vitals saved")
 c=db();rows=c.execute("SELECT * FROM vitals WHERE user_id=? ORDER BY id DESC LIMIT 20",(u["id"],)).fetchall();c.close();return render_template("vitals.html",rows=rows)
@app.route("/appointments",methods=["GET","POST"])
def appointments():
 u=user()
 if not u:return redirect("/login")
 if request.method=="POST":
  c=db();c.execute("INSERT INTO appointments(user_id,doctor,department,date,reason,status,created) VALUES(?,?,?,?,?,'Requested',?)",(u["id"],request.form["doctor"],request.form["department"],request.form["date"],request.form["reason"],datetime.now().isoformat(timespec="seconds")));c.commit();c.close();flash("Appointment request created")
 c=db();rows=c.execute("SELECT * FROM appointments WHERE user_id=? ORDER BY id DESC",(u["id"],)).fetchall();c.close();return render_template("appointments.html",rows=rows)
@app.route("/report",methods=["GET","POST"])
def report():
 u=user()
 if not u:return redirect("/login")
 r=None
 if request.method=="POST":
  uploaded=request.files.get("pdf")
  pasted=request.form.get("text","").strip()
  extracted=pasted
  filename="pasted-report.txt"
  pages=0

  if uploaded and uploaded.filename:
   filename=secure_filename(uploaded.filename)
   if not filename.lower().endswith(".pdf"):
    flash("Only PDF files are allowed.")
    return render_template("report.html",r=None)
   if uploaded.content_length and uploaded.content_length > 10*1024*1024:
    flash("PDF is too large. Maximum size is 10 MB.")
    return render_template("report.html",r=None)
   safe_name=f"{u['id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
   path=os.path.join(UPLOAD_DIR,safe_name)
   uploaded.save(path)
   try:
    reader=PdfReader(path)
    pages=len(reader.pages)
    extracted="\n".join((page.extract_text() or "") for page in reader.pages)[:20000]
   except Exception:
    flash("The PDF could not be read. Please upload a valid text-based PDF.")
    return render_template("report.html",r=None)

  t=extracted.lower()
  rules={
   "high glucose":["high glucose","elevated glucose","hyperglycemia"],
   "low hemoglobin":["low hemoglobin","low hb","anemia"],
   "high blood pressure":["hypertension","high blood pressure"],
   "low oxygen":["low spo2","low oxygen","hypoxia"],
   "infection wording":["infection","elevated wbc","high wbc"]
  }
  flags=[k for k,v in rules.items() if any(q in t for q in v)]
  summary=("Potentially relevant wording detected. Review the highlighted categories with a qualified professional."
           if flags else "No configured warning phrase was detected by this demo analyzer.")
  if not extracted.strip():
   summary="The PDF has no extractable text. It may be a scanned/image-only PDF; OCR would be required for text extraction."
  r={"filename":filename,"pages":pages,"flags":flags,"summary":summary,"text":extracted[:6000]}
  c=db();c.execute("INSERT INTO reports(user_id,text,flags,created) VALUES(?,?,?,?)",
                   (u["id"],extracted[:20000],",".join(flags),datetime.now().isoformat(timespec="seconds")));c.commit();c.close()
 return render_template("report.html",r=r)
@app.route("/api/chat",methods=["POST"])
def chat():
 if not user():return jsonify(error="login required"),401
 m=(request.json or {}).get("message","").lower()
 if any(x in m for x in ["chest pain","unconscious","seizure","stroke","can't breathe","severe bleeding"]):a="This may be an emergency. Seek immediate professional emergency care. This assistant cannot manage emergencies."
 elif "appointment" in m:a="Open Appointments to request a department, doctor and time."
 elif "medicine" in m or "tablet" in m:a="I can provide general medication information, but I cannot prescribe medicines or dosages."
 else:a="I can guide you through AI triage, vitals, appointments and report-text analysis."
 return jsonify(reply=a)
@app.route("/api/health")
def health():return jsonify(status="ok",service="AI HealthCare Assistant",engine="synthetic Naive Bayes triage")
init()
if __name__=="__main__":app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)),debug=True)
